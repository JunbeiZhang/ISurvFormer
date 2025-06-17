import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index


class ColumnwiseImputer(nn.Module):
    """
    Feature-wise LSTM imputer.
    Each feature is predicted using other features at each time point.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.imputers = nn.ModuleList([
            nn.LSTM(input_dim - 1, hidden_dim, num_layers=num_layers, batch_first=True)
            for _ in range(input_dim)
        ])
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(input_dim)])

    def forward(self, x_raw, mask_pad):
        """
        Args:
            x_raw: [B, T, D] original input (NaN-filled for missing values)
            mask_pad: [B, T] padding mask (True=valid, False=pad)
        Returns:
            x_hat: [B, T, D] imputed tensor
        """
        B, T, D = x_raw.shape
        x_hat_list = []

        for d in range(D):
            other_idx = [i for i in range(D) if i != d]
            x_input = x_raw[:, :, other_idx]  # [B, T, D-1]

            # Zero out padded time steps
            mask_broadcast = mask_pad.unsqueeze(-1).expand(-1, -1, D - 1)
            x_input = x_input.clone()
            x_input[~mask_broadcast] = 0

            h, _ = self.imputers[d](x_input)  # [B, T, H]
            pred_d = self.output_layers[d](h)  # [B, T, 1]
            x_hat_list.append(pred_d)

        x_hat = torch.cat(x_hat_list, dim=2)
        return x_hat


class TimeEmbedding(nn.Module):
    """
    Generate time embeddings using either positional encoding or a learnable projection.
    """
    def __init__(self, d_model, method="positional"):
        super().__init__()
        self.d_model = d_model
        self.method = method
        if method == "learnable":
            self.linear = nn.Linear(1, d_model)

    def forward(self, t):
        if self.method == "learnable":
            return self.linear(t)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / self.d_model)).to(t.device)
        pe = torch.zeros(t.size(0), t.size(1), self.d_model).to(t.device)
        pe[:, :, 0::2] = torch.sin(t * div_term)
        pe[:, :, 1::2] = torch.cos(t * div_term)
        return pe


def get_activation(name):
    return {
        'ReLU': nn.ReLU(),
        'GELU': nn.GELU(),
        'SELU': nn.SELU(),
        'LeakyReLU': nn.LeakyReLU()
    }[name]


def make_mlp_head(input_dim, output_dim, hidden_dim, num_layers, activation):
    act_fn = get_activation(activation)
    layers = [nn.Linear(input_dim, hidden_dim), act_fn]
    for _ in range(num_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), act_fn]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


class DynamicSurvTransformer(nn.Module):
    """
    Cluster-aware dynamic survival transformer with latent cluster inference via soft assignment.
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1,
                 time_embed="positional", mlp_hidden_ratio=1.0, mlp_num_layers=2, activation="ReLU",
                 risk_hidden_dim=64, risk_num_layers=2, risk_activation="ReLU", num_clusters=2):
        super().__init__()
        hidden_dim = int(d_model * mlp_hidden_ratio)
        act_fn = get_activation(activation)

        layers = [nn.Linear(input_dim, hidden_dim), act_fn]
        for _ in range(mlp_num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn])
        layers.append(nn.Linear(hidden_dim, d_model))

        self.imputer = ColumnwiseImputer(input_dim=input_dim)
        self.input_mlp = nn.Sequential(*layers)
        self.time_embed = TimeEmbedding(d_model, method=time_embed)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.num_clusters = num_clusters
        self.risk_heads = nn.ModuleList([
            make_mlp_head(d_model, 1, risk_hidden_dim, risk_num_layers, risk_activation)
            for _ in range(num_clusters)
        ])
        self.cluster_proj = nn.Linear(d_model, num_clusters)
        self.tte_head = nn.Linear(d_model, 1)

    def forward(self, x_raw, mask_raw, t, mask_pad):
        """
        Forward pass for ISurvFormer-C.

        Args:
            x_raw: [B, T, D] input with NaN set to 0
            mask_raw: [B, T, D] missing mask (1=observed, 0=missing)
            t: [B, T, 1] timestamp
            mask_pad: [B, T] padding mask

        Returns:
            h_seq: hidden states
            risk_seq: expected risk scores
            tte_seq: predicted time-to-event
            x_filled: imputed input
            q_probs: cluster assignment probabilities
        """
        x_filled = self.imputer(x_raw, mask_pad)

        x_embed = self.input_mlp(x_filled)
        t_embed = self.time_embed(t)
        src = x_embed + t_embed

        src_key_padding_mask = ~mask_pad if mask_pad is not None else None
        h_seq = self.transformer(src, src_key_padding_mask=src_key_padding_mask)

        tte_seq = self.tte_head(h_seq).squeeze(-1)

        # Cluster projection q(c|z)
        z = h_seq[:, -1, :]  # use last time point
        q_logits = self.cluster_proj(z)
        q_probs = F.softmax(q_logits, dim=-1)  # [B, K]

        # Cluster-specific risk heads
        B, T, _ = h_seq.shape
        risk_all = [head(h_seq).squeeze(-1) for head in self.risk_heads]
        risk_all = torch.stack(risk_all, dim=-1)  # [B, T, K]

        q_probs_exp = q_probs.unsqueeze(1).expand(B, T, self.num_clusters)
        risk_seq = (risk_all * q_probs_exp).sum(dim=2)  # expected risk

        return h_seq, risk_seq, tte_seq, x_filled, q_probs


def cox_partial_log_likelihood(risk, durations, events):
    """
    Compute negative Cox partial log-likelihood.
    """
    order = torch.argsort(durations)
    risk, durations, events = risk[order], durations[order], events[order]
    exp_risk = torch.exp(risk)
    cum_sum = torch.flip(torch.cumsum(torch.flip(exp_risk, dims=[0]), dim=0), dims=[0])
    return -(risk[events == 1] - torch.log(cum_sum[events == 1] + 1e-8)).sum()


def kl_categorical_uniform(q_probs):
    """
    KL divergence between q(c|z) and uniform prior.
    """
    B, K = q_probs.size()
    log_q = torch.log(q_probs + 1e-8)
    log_uniform = torch.log(torch.tensor(1.0 / K)).to(q_probs.device)
    return (q_probs * (log_q - log_uniform)).sum(dim=1).mean()


def dynamic_survival_loss_with_imputation(
    q_probs, eta, tte_seq, durations, events, lengths, times,
    x_filled, x_true, mask_raw, alpha=1.0, beta=1.0, gamma=1.0
):
    """
    Total loss: Cox + TTE + Imputation + KL(cluster)
    """
    B, L, D = x_true.shape
    device = x_true.device

    loss_cox = cox_partial_log_likelihood(eta, durations, events)

    durations = durations.unsqueeze(1).expand(-1, L)
    true_remain_time = durations - times
    valid_mask = (torch.arange(L).expand(B, L).to(device) < lengths.unsqueeze(1)) & (events.unsqueeze(1) == 1)
    loss_tte = F.mse_loss(tte_seq[valid_mask], true_remain_time[valid_mask])

    mask_missing = (mask_raw == 0)
    loss_impute = F.mse_loss(x_filled[mask_missing], x_true[mask_missing]) if mask_missing.sum() > 0 else torch.tensor(0.0, device=device)

    loss_kl = kl_categorical_uniform(q_probs)

    return loss_cox + alpha * loss_tte + beta * loss_impute + gamma * loss_kl


def generate_mask(lengths, max_len=None):
    """
    Generate padding mask from sequence lengths.
    """
    B = lengths.size(0)
    max_len = lengths.max().item() if max_len is None else max_len
    idxs = torch.arange(max_len).expand(B, max_len).to(lengths.device)
    return idxs < lengths.unsqueeze(1)


def dynamic_cindex(risk_seq, durations, events, lengths):
    """
    Compute time-wise C-index.
    """
    B, L = risk_seq.shape
    cindex_list = []
    for t in range(L):
        mask_t = (torch.arange(L).to(lengths.device)[t] < lengths) & (events == 1)
        if mask_t.sum() < 2:
            continue
        cidx = concordance_index(
            durations[mask_t].cpu().numpy(),
            -risk_seq[mask_t, t].detach().cpu().numpy(),
            events[mask_t].cpu().numpy()
        )
        cindex_list.append(cidx)
    return cindex_list
