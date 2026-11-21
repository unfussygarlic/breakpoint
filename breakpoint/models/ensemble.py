"""Stage G: a dynamics ensemble with aleatoric + epistemic uncertainty.

Each member is a small MLP with two heads: a mean Delta-s and a log-variance
(aleatoric). Members differ by random init and minibatch order, so their
DISAGREEMENT estimates epistemic uncertainty. Trained by Gaussian NLL, with an
MSE warm-up so the mean fits before the variance head engages.

Total predictive variance = mean of member variances (aleatoric)
                          + variance of member means  (epistemic)

Works on standardized (s,a)->Delta-s. Device = CUDA if available else CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPMember(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, out_dim)
        self.logvar = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.body(x)
        return self.mean(h), self.logvar(h).clamp(-8.0, 4.0)


def gaussian_nll(mean, logvar, target):
    inv = torch.exp(-logvar)
    return (0.5 * (logvar + (target - mean) ** 2 * inv)).sum(-1).mean()


class DynamicsEnsemble:
    def __init__(self, in_dim: int = 6, out_dim: int = 4, n_members: int = 5,
                 hidden: int = 64, seed: int = 0):
        self.n = n_members
        self.members = []
        for k in range(n_members):
            torch.manual_seed(seed * 100 + k)
            self.members.append(MLPMember(in_dim, out_dim, hidden).to(DEVICE))

    def train(self, X: np.ndarray, D: np.ndarray, epochs: int = 150, warmup: int = 50,
              batch: int = 256, lr: float = 1e-3, seed: int = 0, verbose: bool = False):
        """First `warmup` epochs fit the MEAN with MSE (avoids NLL variance
        inflation), then switch to Gaussian NLL for the variance head."""
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        Dt = torch.tensor(D, dtype=torch.float32, device=DEVICE)
        n = len(Xt)
        for k, m in enumerate(self.members):
            g = torch.Generator(device="cpu").manual_seed(seed * 100 + k)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            m.train()
            for ep in range(epochs):
                perm = torch.randperm(n, generator=g).to(DEVICE)
                for i in range(0, n, batch):
                    idx = perm[i:i + batch]
                    mean, logvar = m(Xt[idx])
                    if ep < warmup:
                        loss = ((mean - Dt[idx]) ** 2).sum(-1).mean()
                    else:
                        loss = gaussian_nll(mean, logvar, Dt[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            if verbose:
                print(f"  member {k}: final loss {loss.item():.4f}")

    @torch.no_grad()
    def predict(self, X: np.ndarray):
        """Per-member means + aggregated uncertainty, in STANDARDIZED target space."""
        Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        means, varis = [], []
        for m in self.members:
            m.eval()
            mu, logvar = m(Xt)
            means.append(mu.cpu().numpy())
            varis.append(torch.exp(logvar).cpu().numpy())
        means = np.stack(means); varis = np.stack(varis)   # (K, N, 4)
        mean = means.mean(0)
        aleatoric = varis.mean(0)
        epistemic = means.var(0)
        return {"mean": mean, "aleatoric": aleatoric, "epistemic": epistemic,
                "total": aleatoric + epistemic, "member_means": means}