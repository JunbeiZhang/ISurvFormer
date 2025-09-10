import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index

class ColumnwiseImputer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.imputers = nn.ModuleList([
            nn.LSTM(input_dim - 1, hidden_dim, num_layers=num_layers, batch_first=True)
            for _ in range(input_dim)
        ])
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(input_dim)])

    def forward(self, x_raw, mask_pad, mask_raw=None):
        B, T, D = x_raw.shape
        x_hat_list = []
        for d in range(D):
            other_idx = [i for i in range(D) if i != d]
            x_input = x_raw[:, :, other_idx].clone()  # [B,T,D-1]

            # 屏蔽 padding
            x_input[~mask_pad.unsqueeze(-1).expand(-1, -1, len(other_idx))] = 0

            # 屏蔽“其他列”的缺失，防信息泄露
            if mask_raw is not None:
                mask_other = mask_raw[:, :, other_idx]  # True=观测
                x_input[~mask_other] = 0

            h, _ = self.imputers[d](x_input)
            pred_d = self.output_layers[d](h)          # [B,T,1]
            x_hat_list.append(pred_d)
        x_hat = torch.cat(x_hat_list, dim=2)           # [B,T,D]
        return x_hat

class TimeEmbedding(nn.Module):
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
        x_raw: [B, T, D] 原始输入（0填充缺失值）
        mask_raw: [B, T, D] 缺失值掩码（1=观测，0=缺失）
        t: [B, T, 1] 时间戳
        mask_pad: [B, T] pad 掩码（时间维度上的）
        """
        x_filled = self.imputer(x_raw, mask_pad, mask_raw=mask_raw)

        x_embed = self.input_mlp(x_filled)
        t_embed = self.time_embed(t)
        src = x_embed + t_embed

        src_key_padding_mask = ~mask_pad if mask_pad is not None else None
        h_seq = self.transformer(src, src_key_padding_mask=src_key_padding_mask)

        risk_seq = self.risk_head(h_seq).squeeze(-1)
        tte_seq = self.tte_head(h_seq).squeeze(-1)
        

        return h_seq, risk_seq, tte_seq, x_filled  # 多返回一个 x_filled



def cox_partial_log_likelihood(risk, durations, events):
    order = torch.argsort(durations)
    risk, durations, events = risk[order], durations[order], events[order]
    exp_risk = torch.exp(risk)
    cum_sum = torch.flip(torch.cumsum(torch.flip(exp_risk, dims=[0]), dim=0), dims=[0])
    log_loss = -(risk[events == 1] - torch.log(cum_sum[events == 1] + (1e-8))).sum()
    return log_loss


import torch
import torch.nn.functional as F

def dynamic_survival_loss_with_imputation(
    h_seq: torch.Tensor,
    risk_seq: torch.Tensor,
    tte_seq: torch.Tensor,
    durations: torch.Tensor,   # [B]
    events: torch.Tensor,      # [B] (1=event, 0=censored)
    lengths: torch.Tensor,     # [B] valid time steps per sample
    times: torch.Tensor,       # [B, L] time at each step
    x_filled: torch.Tensor,    # [B, L, D] model-imputed sequence
    x_true: torch.Tensor,      # [B, L, D] ground truth (observed entries)
    mask_raw: torch.Tensor,    # [B, L, D] (1=observed, 0=missing)
    alpha: float = 1.0,        # weight for TTE loss
    beta: float = 1.0          # weight for imputation loss
) -> torch.Tensor:
    """
    Loss = Cox partial log-likelihood (last step) 
           + alpha * TTE regression loss (valid, event=1 steps)
           + beta  * Imputation loss (MSE on observed & non-padding entries).

    Notes:
    - Imputation loss follows the "observed-supervised" design:
      supervise ONLY at positions that are originally observed and within the
      non-padded region; missing and padded positions are not supervised.
    """
    device = x_true.device
    B, L, D = x_true.shape

    # === Cox loss (use the last valid step per sequence) ===
    last_idx = lengths - 1  # [B]
    risk_last = risk_seq[torch.arange(B, device=device), last_idx]  # [B]
    loss_cox = cox_partial_log_likelihood(risk_last, durations, events)

    # === TTE regression loss (only for non-padding steps of event samples) ===
    # true remaining time = (event/censor time) - current time
    durations_exp = durations.unsqueeze(1).expand(-1, L)  # [B, L]
    true_remain_time = durations_exp - times              # [B, L]
    valid_time_mask = (torch.arange(L, device=device).view(1, L) < lengths.view(B, 1))  # [B, L]
    event_mask = (events.view(B, 1) == 1)                                        # [B, L]
    tte_mask = valid_time_mask & event_mask                                      # [B, L]

    if tte_mask.any():
        loss_tte = F.mse_loss(tte_seq[tte_mask], true_remain_time[tte_mask])
    else:
        loss_tte = torch.tensor(0.0, device=device)

    # === Imputation loss (ONLY on observed & non-padding positions) ===
    # Build non-padding mask in 3D to match feature dimension
    non_pad_3d = (torch.arange(L, device=device).view(1, L, 1) < lengths.view(B, 1, 1))  # [B, L, 1]
    obs_mask = (mask_raw.to(torch.bool)) & non_pad_3d                                    # [B, L, D]

    if obs_mask.any():
        loss_impute = F.mse_loss(x_filled[obs_mask], x_true[obs_mask])
    else:
        loss_impute = torch.tensor(0.0, device=device)

    # === Total ===
    return loss_cox + alpha * loss_tte + beta * loss_impute



def generate_mask(lengths, max_len=None):
    B = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()
    idxs = torch.arange(max_len).expand(B, max_len).to(lengths.device)
    return idxs < lengths.unsqueeze(1)


def dynamic_cindex(risk_seq, durations, events, lengths):
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
