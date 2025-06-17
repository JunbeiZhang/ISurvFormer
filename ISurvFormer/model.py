import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index


class ColumnwiseImputer(nn.Module):
    """
    LSTM-based feature-wise imputer.
    Each feature is predicted from all other features using an LSTM.
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
            x_raw (Tensor): [B, T, D] original input (missing entries should be NaN)
            mask_pad (Tensor): [B, T] padding mask (True = valid time steps)

        Returns:
            x_hat (Tensor): [B, T, D] imputed tensor
        """
        B, T, D = x_raw.shape
        x_hat_list = []

        for d in range(D):
            other_idx = [i for i in range(D) if i != d]
            x_input = x_raw[:, :, other_idx]  # [B, T, D-1]

            # Mask out padded time steps
            mask_broadcast = mask_pad.unsqueeze(-1).expand(-1, -1, D - 1)
            x_input = x_input.clone()
            x_input[~mask_broadcast] = 0

            h, _ = self.imputers[d](x_input)
            pred_d = self.output_layers[d](h)
            x_hat_list.append(pred_d)

        x_hat = torch.cat(x_hat_list, dim=2)  # [B, T, D]
        return x_hat


class TimeEmbedding(nn.Module):
    """
    Positional or learnable time embedding for input timestamps.
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
        position = t
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / self.d_model)).to(t.device)
        pe = torch.zeros(t.size(0), t.size(1), self.d_model).to(t.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
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
    Transformer-based survival model with built-in imputation and time embedding.
    """

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1,
                 time_embed="positional", mlp_hidden_ratio=1.0, mlp_num_layers=2, activation="ReLU",
                 risk_hidden_dim=64, risk_num_layers=2, risk_activation="ReLU"):
        super().__init__()
        hidden_dim = int(d_model * mlp_hidden_ratio)
        act_fn = get_activation(activation)

        layers = [nn.Linear(input_dim, hidden_dim), act_fn]
        for _ in range(mlp_num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)
        layers.append(nn.Linear(hidden_dim, d_model))

        self.imputer = ColumnwiseImputer(input_dim=input_dim)
        self.input_mlp = nn.Sequential(*layers)
        self.time_embed = TimeEmbedding(d_model, method=time_embed)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.risk_head = make_mlp_head(
            input_dim=d_model,
            output_dim=1,
            hidden_dim=risk_hidden_dim,
            num_layers=risk_num_layers,
            activation=risk_activation
        )

        self.tte_head = nn.Linear(d_model, 1)

    def forward(self, x_raw, mask_raw, t, mask_pad):
        """
        Args:
            x_raw (Tensor): [B, T, D] input with NaN replaced by 0
            mask_raw (Tensor): [B, T, D] missing mask (1 = observed, 0 = missing)
            t (Tensor): [B, T, 1] timestamps
            mask_pad (Tensor): [B, T] padding mask (True = valid)

        Returns:
            h_seq (Tensor): hidden states from Transformer [B, T, d_model]
            risk_seq (Tensor): risk prediction sequence [B, T]
            tte_seq (Tensor): predicted remaining time-to-event [B, T]
            x_filled (Tensor): imputed inputs [B, T, D]
        """
        x_filled = self.imputer(x_raw, mask_pad)

        x_embed = self.input_mlp(x_filled)
        t_embed = self.time_embed(t)
        src = x_embed + t_embed

        src_key_padding_mask = ~mask_pad if mask_pad is not None else None
        h_seq = self.transformer(src, src_key_padding_mask=src_key_padding_mask)

        risk_seq = self.risk_head(h_seq).squeeze(-1)
        tte_seq = self.tte_head(h_seq).squeeze(-1)

        return h_seq, risk_seq, tte_seq, x_filled


def cox_partial_log_likelihood(risk, durations, events):
    """
    Computes negative partial log-likelihood for Cox proportional hazards model.
    """
    order = torch.argsort(durations)
    risk, durations, events = risk[order], durations[order], events[order]
    exp_risk = torch.exp(risk)
    cum_sum = torch.flip(torch.cumsum(torch.flip(exp_risk, dims=[0]), dim=0), dims=[0])
    log_loss = -(risk[events == 1] - torch.log(cum_sum[events == 1] + 1e-8)).sum()
    return log_loss


def dynamic_survival_loss_with_imputation(h_seq, risk_seq, tte_seq, durations, events, lengths, times,
                                          x_filled, x_true, mask_raw, alpha=1.0, beta=1.0):
    """
    Combined loss function for survival prediction and imputation.
    Includes:
        - Cox partial likelihood loss
        - MSE for time-to-event regression
        - MSE for imputation on missing values
    """
    B, L, D = x_true.shape
    device = x_true.device

    last_idx = lengths - 1
    risk_last = risk_seq[torch.arange(B), last_idx]
    loss_cox = cox_partial_log_likelihood(risk_last, durations, events)

    durations = durations.unsqueeze(1).expand(-1, L)
    true_remain_time = durations - times
    valid_mask = (torch.arange(L).expand(B, L).to(device) < lengths.unsqueeze(1)) & (events.unsqueeze(1) == 1)
    loss_tte = F.mse_loss(tte_seq[valid_mask], true_remain_time[valid_mask])

    mask_missing = (mask_raw == 0)
    if mask_missing.sum() > 0:
        loss_impute = F.mse_loss(x_filled[mask_missing], x_true[mask_missing])
    else:
        loss_impute = torch.tensor(0.0, device=device)

    return loss_cox + alpha * loss_tte + beta * loss_impute


def generate_mask(lengths, max_len=None):
    """
    Generates a boolean mask for each sequence based on its valid length.
    Returns a tensor of shape [B, T] with True indicating valid time steps.
    """
    B = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()
    idxs = torch.arange(max_len).expand(B, max_len).to(lengths.device)
    return idxs < lengths.unsqueeze(1)


def dynamic_cindex(risk_seq, durations, events, lengths):
    """
    Calculates C-index at each time step in the risk sequence.
    """
    B, L = risk_seq.shape
    cindex_list = []
    for t in range(L):
        mask_t = (torch.arange(L).to(lengths.device)[t] < lengths) & (events == 1)
        if mask_t.sum() < 2:
            continue
        cidx = concordance_index(durations[mask_t].cpu().numpy(),
                                 -risk_seq[mask_t, t].detach().cpu().numpy(),
                                 events[mask_t].cpu().numpy())
        cindex_list.append(cidx)
    return cindex_list
