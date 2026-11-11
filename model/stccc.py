"""stCCC: spatial VAE with spatially-gated, communication-biased neighbour attention.

Changes relative to the AAAI-27 submission (Spatial-ccc-gb/gatingUnit/SpatialCCC_v2_gating.py):
  1. Neighbour attention is inductive. Keys/values are the gated latent means of each spot's spatial
     kNN neighbours, not free per-spot embeddings, so inference does not depend on batch composition.
  2. Communication lives on a fixed spatial kNN graph (O(Nk)). Every spot keeps k partners in every
     mini-batch, independent of N, so the regulariser no longer thins out under random batching.
  3. The consistency loss is a per-spot KL between the communication distribution over neighbours and
     the latent-similarity distribution over the same neighbours. Both sides are distributions on the
     same support (the old loss compared a [0,1] rank with a row-softmax over the whole batch), and the
     loss is summed over spots, the same scale convention as the reconstruction and KL terms.
  4. Edge scores are computed on library-normalised expression and depth-residualised (ccc_graph.py).
"""
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset, random_split

from I_PID import PIDControl
from SVGP import SVGP
from VAE_utils import DenseEncoder, MeanAct, NBLoss, buildNetwork, gauss_cross_entropy


class SpatialGatingUnit(nn.Module):
    """Per-spot convex weights [w_expression, w_spatial] from encoder hidden state and coordinates."""
    def __init__(self, hidden_dim, spatial_dim=2):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_dim + spatial_dim, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, h, x):
        return F.softmax(self.gate(torch.cat([h, x], -1)), -1)


class NeighbourAttention(nn.Module):
    """Attention of each spot over its k spatial neighbours.
    logit_ij = <q(mu_i), k(mu_j)>/sqrt(d) + gamma * s_ij + log M_ij,
    where s_ij is the (z-scored) communication score and M_ij the Gaussian spatial kernel."""
    def __init__(self, dim, att_dim=32):
        super().__init__()
        self.q = nn.Linear(dim, att_dim); self.k = nn.Linear(dim, att_dim); self.v = nn.Linear(dim, dim)
        self.scale = att_dim ** -0.5

    def forward(self, mu_i, mu_nbr, logM, score=None, gamma=1.0):
        # mu_i (B, D), mu_nbr (B, k, D), logM (B, k), score (B, k) or None
        logits = torch.einsum('bd,bkd->bk', self.q(mu_i), self.k(mu_nbr)) * self.scale + logM
        if score is not None:
            logits = logits + gamma * score
        a = F.softmax(logits, -1)
        return mu_i + torch.einsum('bk,bkd->bd', a, self.v(mu_nbr)), a


class stCCC(nn.Module):
    def __init__(self, input_dim, GP_dim, Normal_dim, encoder_layers, decoder_layers, inducing_points,
                 kernel_scale, N_train, nbr, nbr_dist, score=None, sigma_sp=5.0,
                 use_attention=True, gamma=1.0, lam=10.0, tau=1.0, use_gating=True,
                 KL_loss=0.025, init_beta=10, min_beta=4, max_beta=25, dtype=torch.float64, device='cuda'):
        super().__init__()
        torch.set_default_dtype(dtype)
        self.dtype, self.device = dtype, device
        self.GP_dim, self.Normal_dim = GP_dim, Normal_dim
        self.KL_loss, self.beta = KL_loss, init_beta
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.svgp = SVGP(fixed_inducing_points=True, initial_inducing_points=inducing_points, fixed_gp_params=False,
                         kernel_scale=kernel_scale, jitter=1e-6, N_train=N_train, dtype=dtype, device=device)
        self.encoder = DenseEncoder(input_dim=input_dim, hidden_dims=encoder_layers, output_dim=GP_dim + Normal_dim,
                                    activation="elu", dropout=0)
        self.decoder = buildNetwork([GP_dim + Normal_dim] + decoder_layers, activation="elu", dropout=0)
        self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), MeanAct())
        self.dec_disp = nn.Parameter(torch.randn(input_dim))
        self.NB_loss = NBLoss()
        self.gating = SpatialGatingUnit(encoder_layers[-1])
        self.spatial_to_normal = nn.Linear(2, Normal_dim)
        self.use_attention = use_attention
        self.use_gating = use_gating        # False: residual mean = encoder mean (gating ablation; modules still built so that init is unchanged)
        self.attn = NeighbourAttention(Normal_dim) if use_attention else None
        self.gamma, self.lam, self.tau = gamma, lam, tau
        # graph tensors
        self.nbr = torch.as_tensor(nbr, dtype=torch.long, device=device)
        self.logM = torch.as_tensor(-nbr_dist ** 2 / (2 * sigma_sp ** 2), dtype=dtype, device=device)
        self.score = None if score is None else torch.as_tensor(score, dtype=dtype, device=device)
        # communication target distribution over neighbours
        self.logP = None if score is None else F.log_softmax(self.score / tau, -1)
        self.to(device)

    # ----- encoder pieces -----
    def gated_mean(self, x, y):
        qmu, qvar = self.encoder(y)
        if not self.use_gating:
            return qmu[:, :self.GP_dim], qvar[:, :self.GP_dim], qmu[:, self.GP_dim:], qvar[:, self.GP_dim:]
        w = self.gating(self.encoder.layers(y), x)
        mu_n = w[:, :1] * qmu[:, self.GP_dim:] + w[:, 1:2] * self.spatial_to_normal(x)
        return qmu[:, :self.GP_dim], qvar[:, :self.GP_dim], mu_n, qvar[:, self.GP_dim:]

    def neighbour_terms(self, idx, X_all, Y_all, mu_n_i):
        """Gated means of the k neighbours of each spot in idx, plus attention output and L_CCC."""
        nb = self.nbr[idx]                               # (B, k)
        flat = nb.reshape(-1)
        _, _, mu_nb, _ = self.gated_mean(X_all[flat], Y_all[flat])
        mu_nb = mu_nb.reshape(nb.shape[0], nb.shape[1], -1)
        logM = self.logM[idx]
        s = None if self.score is None else self.score[idx]
        mu_enh = mu_n_i
        if self.use_attention:
            mu_enh, _ = self.attn(mu_n_i, mu_nb, logM, s, self.gamma)
        l_ccc = torch.zeros((), dtype=self.dtype, device=self.device)
        if self.logP is not None and self.lam > 0:
            logQ = F.log_softmax(-((mu_n_i[:, None, :] - mu_nb) ** 2).sum(-1), -1)
            logP = self.logP[idx]
            l_ccc = (logP.exp() * (logP - logQ)).sum()   # sum_i KL(P_i || Q_i)
        return mu_enh, l_ccc

    def gp_posterior(self, x, gp_mu, gp_var):
        m, v, rec, kl = [], [], [], []
        for l in range(self.GP_dim):
            pm, pv, mh, Ah = self.svgp.approximate_posterior_params(x, x, gp_mu[:, l], gp_var[:, l])
            r, k = self.svgp.variational_loss(x=x, y=gp_mu[:, l], noise=gp_var[:, l], mu_hat=mh, A_hat=Ah)
            m.append(pm); v.append(pv); rec.append(r); kl.append(k)
        return torch.stack(m, 1), torch.stack(v, 1), torch.stack(rec).sum(), torch.stack(kl).sum()

    def forward(self, idx, x, y, raw_y, sf, X_all, Y_all):
        self.train()
        b = y.shape[0]
        gp_mu, gp_var, mu_n, var_n = self.gated_mean(x, y)
        gp_pm, gp_pv, in_rec, in_kl = self.gp_posterior(x, gp_mu, gp_var)
        inside_elbo = in_rec - (b / self.svgp.N_train) * in_kl
        gp_KL = gauss_cross_entropy(gp_pm, gp_pv, gp_mu, gp_var).sum() - inside_elbo
        mu_enh, l_ccc = self.neighbour_terms(idx, X_all, Y_all, mu_n)
        g_KL = kl_divergence(Normal(mu_enh, torch.sqrt(var_n)), Normal(torch.zeros_like(mu_enh), torch.ones_like(var_n))).sum()
        z = Normal(torch.cat([gp_pm, mu_enh], 1), torch.sqrt(torch.cat([gp_pv, var_n], 1))).rsample()
        mean = self.dec_mean(self.decoder(z))
        disp = torch.exp(torch.clamp(self.dec_disp, -15., 15.)).unsqueeze(0)
        recon = self.NB_loss(x=raw_y, mean=mean, disp=disp, scale_factor=sf)
        loss = recon + self.beta * (gp_KL + g_KL) + self.lam * l_ccc
        return loss, recon, gp_KL, g_KL, l_ccc

    @torch.no_grad()
    def gate_weights(self, X_all, Y_all, batch_size=512):
        """Spatial gating weight omega^s of every spot (N,): the share of the residual latent mean taken from position."""
        self.eval(); out = []
        for i in range(0, X_all.shape[0], batch_size):
            x, y = X_all[i:i + batch_size], Y_all[i:i + batch_size]
            out.append(self.gating(self.encoder.layers(y), x)[:, 1].cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def gate_terms(self, X_all, Y_all, batch_size=512):
        """Both terms of the gated residual mean, omega^e * mu_expr and omega^s * Psi(x), each (N, Normal_dim)."""
        self.eval(); E, S = [], []
        for i in range(0, X_all.shape[0], batch_size):
            x, y = X_all[i:i + batch_size], Y_all[i:i + batch_size]
            qmu, _ = self.encoder(y); w = self.gating(self.encoder.layers(y), x)
            E.append((w[:, :1] * qmu[:, self.GP_dim:]).cpu()); S.append((w[:, 1:2] * self.spatial_to_normal(x)).cpu())
        return torch.cat(E).numpy(), torch.cat(S).numpy()

    @torch.no_grad()
    def denoise(self, X_all, Y_all, batch_size=512):
        """Decoder mean at the posterior latent mean: NB rate per unit size factor (library-free), (N, G)."""
        z = torch.as_tensor(self.latent(X_all, Y_all, batch_size), dtype=self.dtype, device=self.device)
        return torch.cat([self.dec_mean(self.decoder(z[s:s + batch_size])) for s in range(0, len(z), batch_size)]).cpu().numpy()

    @torch.no_grad()
    def latent(self, X_all, Y_all, batch_size=512):
        self.eval()
        N = X_all.shape[0]; out = []
        for s in range(0, N, batch_size):
            idx = torch.arange(s, min(s + batch_size, N), device=self.device)
            x, y = X_all[idx], Y_all[idx]
            gp_mu, gp_var, mu_n, _ = self.gated_mean(x, y)
            gp_pm = torch.stack([self.svgp.approximate_posterior_params(x, x, gp_mu[:, l], gp_var[:, l])[0]
                                 for l in range(self.GP_dim)], 1)
            mu_enh, _ = self.neighbour_terms(idx, X_all, Y_all, mu_n)
            out.append(torch.cat([gp_pm, mu_enh], 1).cpu())
        return torch.cat(out).numpy()

    def fit(self, pos, ncounts, raw_counts, size_factors, lr=1e-3, weight_decay=1e-6, batch_size=256,
            train_size=0.95, maxiter=100, patience=50, min_steps=1200, verbose=False, callback=None):
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)
        X_all, Y_all = T(pos), T(ncounts)
        ds = TensorDataset(torch.arange(len(pos)), T(raw_counts).cpu(), T(size_factors).cpu())
        n_tr = int(round(train_size * len(ds)))
        tr, va = random_split(ds, [n_tr, len(ds) - n_tr])
        # Training budget in gradient steps: at least min_steps (= 100 epochs on the smallest DLPFC section, so DLPFC and
        # MERFISH training is unchanged); small sections (HER2 ST, a few hundred spots) get more epochs instead of
        # fewer steps, and a section smaller than one batch is trained in full batches instead of not at all.
        batch_size = min(batch_size, n_tr)
        per_ep = max(n_tr // batch_size, 1)
        scale = max(1.0, min_steps / (maxiter * per_ep))
        maxiter, patience = int(np.ceil(maxiter * scale)), int(np.ceil(patience * scale))
        dl = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True)
        vl = DataLoader(va, batch_size=batch_size, shuffle=False)
        opt = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        q = deque(); best, wait = np.inf, 0; best_state = None
        self.history = []
        for ep in range(maxiter):
            agg = np.zeros(5); n = 0
            for idx, raw, sf in dl:
                idx = idx.to(self.device)
                loss, rec, gkl, nkl, lc = self.forward(idx, X_all[idx], Y_all[idx], raw.to(self.device), sf.to(self.device), X_all, Y_all)
                opt.zero_grad(); loss.backward(); opt.step()
                agg += np.array([loss.item(), rec.item(), gkl.item(), nkl.item(), lc.item()]); n += len(idx)
                q.append((gkl.item() + nkl.item()) / len(idx))
                self.beta, _ = self.PID.pid(self.KL_loss * (self.GP_dim + self.Normal_dim), np.mean(q))
                if len(q) >= 10: q.popleft()
            with torch.no_grad():
                vloss, vn = 0.0, 0
                for idx, raw, sf in vl:
                    idx = idx.to(self.device)
                    vloss += self.forward(idx, X_all[idx], Y_all[idx], raw.to(self.device), sf.to(self.device), X_all, Y_all)[0].item(); vn += len(idx)
            self.history.append(dict(epoch=ep + 1, **dict(zip(['loss', 'recon', 'gpKL', 'nKL', 'L_ccc'], agg / n)), val=vloss / vn, beta=self.beta))
            if verbose: print(self.history[-1], flush=True)
            if callback is not None: callback(self, ep)
            if vloss / vn < best:
                best, wait = vloss / vn, 0
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            else:
                wait += 1
                if wait >= patience: break
        if best_state is not None:
            self.load_state_dict(best_state)
        return self
