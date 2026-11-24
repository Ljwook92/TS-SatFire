"""Train a perimeter residual head and evaluate paired full-crop rolling forecasts."""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from directions import Directions, corrected_scores
from encoder import Encoder

ASSETS = Path(__file__).resolve().parent / 'assets'
NM = 'US_2021_NM3323810847220210520'
THRESHOLDS = np.round(np.arange(-8, 6.01, .1), 2)


def unpack(a):
    return np.unpackbits(a).reshape(256, 256).astype(bool)


def make_bases(rows, cache, dest, ts, device):
    encoder = Encoder().to(device).eval()
    encoder.load_state_dict(torch.load(ASSETS / f'encoder_ts{ts}.pt', map_location=device, weights_only=True))
    with torch.inference_mode():
        for r in rows.itertuples():
            path = dest / r.event / f'{r.target_index:03d}.npy'
            if path.is_file():
                continue
            path.parent.mkdir(exist_ok=True, parents=True)
            seq = []
            for i in range(r.target_index-ts, r.target_index):
                with np.load(cache / r.event / f'{i:03d}.npz') as z:
                    seq.append(z['enc'])
            x = torch.from_numpy(np.stack(seq)).unsqueeze(0).to(device, torch.float32)
            _, base = encoder(x)
            tmp = path.with_suffix('.tmp.npy')
            np.save(tmp, base[0].cpu().numpy())
            tmp.replace(path)


class Windows(Dataset):
    def __init__(self, rows, cache, bases, ts):
        self.rows = list(rows.itertuples(index=False))
        self.cache, self.bases, self.ts = cache, bases, ts

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        seq = []
        for i in range(r.target_index-self.ts, r.target_index):
            with np.load(self.cache / r.event / f'{i:03d}.npz') as z:
                frame = z['local']
                if not seq:
                    assert str(z['date']) == r.first_input
                    frame[[3, 4, 5, 6]] = 0  # first-day thermal change extends outside this window
                seq.append(frame)
                if i == r.target_index-1:
                    assert str(z['date']) == r.last_input
                    past, recent, owner = unpack(z['past']), unpack(z['new']), z['owner']
        with np.load(self.cache / r.event / f'{r.target_index:03d}.npz') as z:
            assert str(z['date']) == r.target_date
            truth = unpack(z['new'])
        assert not (truth & past).any() and truth.sum() == r.nan_zero_positives
        base = np.load(self.bases / r.event / f'{r.target_index:03d}.npy')
        return np.concatenate(seq), base, owner, past, recent, truth, idx


class BoundaryHead(nn.Module):
    def __init__(self, ts):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(ts*21+1, 24, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(24, 16, 3, padding=2, dilation=2), nn.SiLU(),
                                 nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, local, base, owner):
        inp = torch.cat([local.float(), base[:, None].clamp(-12, 12)/4], 1)
        adjustment = self.net(inp).squeeze(1)
        return base + adjustment * (owner >= 0)


def boundary_loss(scores, owner, truth):
    keep = owner >= 0
    if not keep.any():
        return scores.sum() * 0
    y, logit = truth[keep].float(), scores[keep]
    pixel = F.binary_cross_entropy_with_logits(logit, y, pos_weight=logit.new_tensor(5.))
    p = logit.sigmoid()
    dice = 1 - (2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
    batch_index = torch.arange(owner.shape[0], device=owner.device)[:, None, None]
    group = (owner.long() + batch_index * (256*256*4))[keep]
    labels, idx = torch.unique(group, return_inverse=True)
    def counts(v):
        return v.new_zeros(len(labels)).scatter_add(0, idx, v)
    extent = F.smooth_l1_loss(counts(p).log1p(), counts(y).log1p())
    return pixel + dice + .5*extent


def hist(score, y):
    bins = np.searchsorted(THRESHOLDS, score, side='right')
    def above(a):
        return np.bincount(a, minlength=len(THRESHOLDS)+1)[::-1].cumsum()[::-1][1:]
    tp = above(bins[y])
    return np.stack([tp, above(bins)-tp, y.sum()-tp])


def confusion(flag, y):
    tp = int((flag & y).sum())
    return tp, int(flag.sum())-tp, int(y.sum())-tp


def evaluate(loader, net, directions, device, dest, choices=None):
    net.eval()
    archived = directions.incumbent
    accum, records, east = {}, [], []
    with torch.inference_mode():
        for local, base, owner, past, recent, truth, indices in loader:
            refined = net(local.to(device), base.to(device), owner.to(device)).cpu().numpy()
            for j, idx in enumerate(indices.tolist()):
                r = loader.dataset.rows[idx]
                old = base[j].numpy().ravel().copy()
                new = refined[j].ravel().copy()
                burned, latest = past[j].numpy().ravel(), recent[j].numpy().ravel()
                old[burned] = new[burned] = -np.inf
                ids, sector, q, source = directions.get(r, burned, latest)
                fields = dict(direct=old, legacy_prior=corrected_scores(old, ids, sector, q, False),
                              allocation_only=corrected_scores(old, ids, sector, q, True),
                              local_extent=corrected_scores(new, ids, sector, q, True))
                y = truth[j].numpy().ravel()
                if choices is None:
                    for name, values in fields.items():
                        accum[name] = accum.get(name, 0) + hist(values, y)
                    continue
                masks = {k: v >= choices[k] for k, v in fields.items()}
                raw_mask = archived.read(f'{r.event}/{r.target_date}.bits')
                masks['archived_ts2'] = np.unpackbits(np.frombuffer(raw_mask, dtype=np.uint8)).astype(bool)
                assert not (masks['archived_ts2'] & burned).any()
                if r.ts == 2:
                    masks['legacy_frozen_ts2'] = fields['legacy_prior'] >= .9
                for name, flag in masks.items():
                    tp, fp, fn = confusion(flag, y)
                    records.append(dict(event=r.event, target=r.target_date, ts=r.ts, model=name,
                                        consecutive=r.calendar_consecutive, prior=source, tp=tp, fp=fp, fn=fn))
                folder = dest / 'masks' / r.event
                folder.mkdir(exist_ok=True, parents=True)
                np.savez_compressed(folder / f'{r.target_date}.npz', **{k: np.packbits(v) for k,v in masks.items()})
                if r.event == NM and r.target_date == '2021-06-01':
                    from report import draw_east
                    east.extend(draw_east(dest, r, masks, y, burned, ids, sector))
    if choices is None:
        result = {}
        for name, (tp, fp, fn) in accum.items():
            ious = tp / np.maximum(tp+fp+fn, 1)
            k = int(ious.argmax())
            result[name] = dict(threshold=float(THRESHOLDS[k]), val_iou=float(ious[k]))
        return result
    pd.DataFrame(records).to_csv(dest/'frame_metrics.csv', index=False)
    pd.DataFrame(east).to_csv(dest/'new_mexico_east.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--ts', type=int, choices=[2,4,6], required=True)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--smoke', action='store_true', help='CPU wiring check on one distinct event per split')
    a = ap.parse_args()
    assert a.epochs >= 1
    torch.manual_seed(804)
    rows = pd.read_csv(ASSETS/'windows.csv.gz', dtype={'event':str})
    assert rows.groupby('event').split.nunique().eq(1).all()
    rows = rows[rows.ts == a.ts].copy()
    assert not rows.duplicated(['event','target_date']).any()
    if a.smoke:
        rows = rows[((rows.event.isin(['20702159','20562846'])) & (rows.target_index == 6)) |
                    ((rows.event == NM) & (rows.target_date == '2021-06-01'))]
        assert len(rows) == 3 and rows.split.nunique() == 3
        device = 'cpu'
        torch.set_num_threads(2)
    else:
        assert torch.cuda.is_available(), 'CUDA GPU required inside Slurm allocation'
        assert all((a.cache/e/'complete').is_file() for e in rows.event.unique())
        device = 'cuda'
    dest = a.run / (('smoke_' if a.smoke else '') + f'TS{a.ts}')
    dest.mkdir(parents=True, exist_ok=False)
    bases = dest/'base'
    train_rows, val_rows, test_rows = [rows[rows.split == k] for k in ('train','val','test')]
    make_bases(pd.concat([train_rows,val_rows]), a.cache, bases, a.ts, device)
    def loader(table, shuffle=False):
        return DataLoader(Windows(table,a.cache,bases,a.ts), batch_size=a.batch, shuffle=shuffle,
                          num_workers=a.workers, persistent_workers=a.workers>0)
    train, val = loader(train_rows,True), loader(val_rows)
    directions = Directions(ASSETS, a.cache)
    directions.incumbent = zipfile.ZipFile(ASSETS/"incumbent_masks.zip")
    net = BoundaryHead(a.ts).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=.0005)
    best = -1
    history = []
    for epoch in range(1,a.epochs+1):
        net.train()
        for x,b,owner,_,_,y,_ in train:
            owner = owner.to(device)
            scores = net(x.to(device), b.to(device), owner)
            loss = boundary_loss(scores, owner, y.to(device))
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            opt.zero_grad()
            loss.backward()
            opt.step()
        result = evaluate(val, net, directions, device, dest)
        history.append(dict(epoch=epoch, models=result))
        print('validation', epoch, result, flush=True)
        if result['local_extent']['val_iou'] > best:
            best = result['local_extent']['val_iou']
            torch.save(net.state_dict(), dest/'best.pt')
            (dest/'choices.json').write_text(json.dumps(history[-1], indent=2))
        (dest/'validation.json').write_text(json.dumps(history, indent=2))
    selected = json.loads((dest/'choices.json').read_text())['models']
    net.load_state_dict(torch.load(dest/'best.pt', map_location=device, weights_only=True))
    make_bases(test_rows, a.cache, bases, a.ts, device)
    evaluate(loader(test_rows), net, directions, device, dest, {k:v['threshold'] for k,v in selected.items()})
    print('finished',dest, flush=True)


if __name__ == '__main__':
    main()
