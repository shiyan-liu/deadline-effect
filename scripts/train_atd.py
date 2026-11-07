"""Archived ATD protocol. Descending-rank AUC matches the reported orientation.
Use symmetric timing AUC for interpretation; no invariance claim is made.
The review split permits paper overlap. Device/precision can change numerical outputs.
"""
import os,json,argparse,math,random,statistics,gzip
from pathlib import Path
from collections import defaultdict
from review_features import extract_paper_features
def adversarial_debiasing(papers: list[dict], epochs: int=50, lr: float=0.001, lambda_adv: float=1.0, emb_dim: int=32, batch_size: int=512, seed: int=42) -> dict:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        import numpy as np
    except ImportError:
        return {'error': '需要 PyTorch 和 NumPy。运行: pip install torch numpy'}
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(os.environ.get('ATD_DEVICE', 'cpu'))
    print(f'  [Path3] device={device}')
    all_ratings = [r['rating'] for p in papers for r in p['reviews']]
    r_min, r_max = (min(all_ratings), max(all_ratings))
    all_gaps = [r['gap_hours'] for p in papers for r in p['reviews']]
    g_max = max(all_gaps) + 0.001
    rows = []
    for paper in papers:
        dec = float(paper['decision_binary'])
        all_pos = sorted([r['relative_pos'] for r in paper['reviews']])
        n_r = len(all_pos)
        q33 = all_pos[n_r // 3]
        for r in paper['reviews']:
            is_late = 1.0 if r['relative_pos'] < q33 else 0.0
            feats = [(r['rating'] - r_min) / (r_max - r_min + 1e-08), r['confidence'] / 5.0, (r.get('soundness', r['rating']) - r_min) / (r_max - r_min + 1e-08), r['gap_hours'] / g_max]
            rows.append((feats, is_late, dec))
    import random as _random
    _random.seed(seed)
    _random.shuffle(rows)
    split = int(len(rows) * 0.8)
    train_rows, test_rows = (rows[:split], rows[split:])

    def to_tensors(data):
        X = torch.tensor([r[0] for r in data], dtype=torch.float32)
        Y_disc = torch.tensor([[r[1]] for r in data], dtype=torch.float32)
        Y_pred = torch.tensor([[r[2]] for r in data], dtype=torch.float32)
        return (X, Y_disc, Y_pred)
    X_tr, Yd_tr, Yp_tr = to_tensors(train_rows)
    X_te, Yd_te, Yp_te = to_tensors(test_rows)
    train_dl = DataLoader(TensorDataset(X_tr, Yd_tr, Yp_tr), batch_size=batch_size, shuffle=True)

    class Encoder(nn.Module):

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, emb_dim))

        def forward(self, x):
            return self.net(x)

    class Head(nn.Module):

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(emb_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

        def forward(self, x):
            return self.net(x)

    class GRL(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            return (-ctx.lam * grad, None)
    encoder = Encoder().to(device)
    predictor = Head().to(device)
    discriminator = Head().to(device)
    opt_enc = optim.Adam(encoder.parameters(), lr=lr, weight_decay=0.0001)
    opt_pred = optim.Adam(predictor.parameters(), lr=lr, weight_decay=0.0001)
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=0.0001)
    bce = nn.BCELoss()
    history = []
    for epoch in range(epochs):
        encoder.train()
        predictor.train()
        discriminator.train()
        ep_pred, ep_disc = ([], [])
        for xb, yd, yp in train_dl:
            xb = xb.to(device)
            yd = yd.to(device)
            yp = yp.to(device)
            emb = encoder(xb)
            opt_disc.zero_grad()
            bce(discriminator(emb.detach()), yd).backward()
            opt_disc.step()
            opt_pred.zero_grad()
            loss_p = bce(predictor(emb.detach()), yp)
            loss_p.backward()
            opt_pred.step()
            ep_pred.append(loss_p.item())
            opt_enc.zero_grad()
            emb2 = encoder(xb)
            lp = bce(predictor(emb2), yp)
            emb2_rev = GRL.apply(emb2, lambda_adv)
            ld = bce(discriminator(emb2_rev), yd)
            (lp + ld).backward()
            opt_enc.step()
            ep_disc.append(ld.item())
        history.append({'epoch': epoch + 1, 'pred_loss': round(sum(ep_pred) / len(ep_pred), 5), 'disc_loss': round(sum(ep_disc) / len(ep_disc), 5)})
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1:3d}: pred_loss={history[-1]['pred_loss']:.4f}  disc_loss(encoder)={history[-1]['disc_loss']:.4f}")
    encoder.eval()
    predictor.eval()
    discriminator.eval()
    with torch.no_grad():
        emb_te = encoder(X_te.to(device))
        pred_p = predictor(emb_te).cpu().numpy().ravel()
        disc_p = discriminator(emb_te).cpu().numpy().ravel()
    Yp_np = Yp_te.numpy().ravel()
    Yd_np = Yd_te.numpy().ravel()

    def auc_np(scores, labels):
        """Wilcoxon-Mann-Whitney AUC（O(n log n)）。"""
        scores = np.array(scores)
        labels = np.array(labels)
        order = np.argsort(-scores)
        labels_s = labels[order]
        n_pos = labels_s.sum()
        n_neg = len(labels_s) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float('nan')
        ranks = np.arange(1, len(labels_s) + 1)
        rank_sum_pos = ranks[labels_s == 1].sum()
        return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    auc_disc_deb = auc_np(disc_p, Yd_np)
    auc_disc_base = auc_np(X_te[:, 3].numpy(), Yd_np)
    paper_preds_deb = defaultdict(list)
    paper_preds_base = defaultdict(list)
    paper_decs = {}
    test_start = split
    paper_counter = 0
    row_idx = 0
    for paper in papers:
        n_r = len(paper['reviews'])
        for local_i in range(n_r):
            global_i = paper_counter * 1 + row_idx
            if row_idx >= test_start:
                te_i = row_idx - test_start
                if 0 <= te_i < len(pred_p):
                    pid = paper['paper_id']
                    paper_preds_deb[pid].append(pred_p[te_i])
                    paper_preds_base[pid].append(float(test_rows[te_i][0][0]))
                    paper_decs[pid] = float(paper['decision_binary'])
            row_idx += 1
    auc_pred_deb = auc_np(pred_p, Yp_np)
    auc_pred_base = auc_np(X_te[:, 0].numpy(), Yp_np)
    if auc_pred_base < 0.5:
        auc_pred_base = 1.0 - auc_pred_base
    if auc_pred_deb < 0.5:
        auc_pred_deb = 1.0 - auc_pred_deb
    if auc_disc_base < 0.5:
        auc_disc_base = 1.0 - auc_disc_base
    auc_disc_deb_sym = 0.5 + abs(auc_disc_deb - 0.5)
    debiasing_success = auc_disc_deb_sym < auc_disc_base and auc_disc_deb_sym < 0.62
    return {'n_train': len(train_rows), 'n_test': len(test_rows), 'epochs': epochs, 'lambda_adv': lambda_adv, 'emb_dim': emb_dim, 'device': str(device), 'architecture': f'Encoder(4→64→{emb_dim}, linear out) + Predictor({emb_dim}→32→1) + Discriminator({emb_dim}→32→1) + GRL(λ={lambda_adv})', 'auc_predictor_baseline': round(auc_pred_base, 4), 'auc_predictor_debiased': round(auc_pred_deb, 4), 'auc_discriminator_baseline': round(auc_disc_base, 4), 'auc_discriminator_debiased': round(auc_disc_deb, 4), 'auc_discriminator_symmetric': round(auc_disc_deb_sym, 4), 'debiasing_success': debiasing_success, 'training_history_last5': history[-5:], 'note': 'Review-level evaluation with paper overlap; symmetric timing AUC is max(AUC, 1-AUC). No independent post-training adversary or invariance claim.'}

if __name__ == '__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--out',default='results/recomputed/atd_rerun.json');args=ap.parse_args()
 root=Path(__file__).resolve().parents[1];raw=json.load(gzip.open(root/'data/iclr2026_numeric.json.gz','rt'));papers=[extract_paper_features(p,0,10) for p in raw];papers=[p for p in papers if p]
 import torch
 torch.set_num_threads(1)
 result=adversarial_debiasing(papers,epochs=args.epochs);dest=root/args.out;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
