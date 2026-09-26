"""
Optional stage 6: cross-encoder re-ranker (Ditto-style) for the ambiguous pairs.

A small pre-trained transformer reads both records as text ("name | address" [SEP] "name | address")
and is fine-tuned to say match / no-match. It is applied only where the GBDT is unsure
(lo <= p <= hi), and its score is blended with the GBDT score:  p = w*p_gbdt + (1-w)*p_ce.
w, the threshold and the decision rule are tuned on the held-out validation entities (which the
cross-encoder never trains on); if the blend does not beat the GBDT alone on validation, w = 1
and the output is identical to stage 5.

Model: intfloat/multilingual-e5-small (MIT licence, ~118M parameters, far below the 8B cap).
"""
import os, gc, json, time, pickle, zlib
import numpy as np
import pandas as pd

from utils import log, read_tsv
from decide import exclusive, by_threshold, by_expected_f, to_sets
from metric import macro_f05

MODEL_NAME = os.environ.get('ER_CE_MODEL', 'intfloat/multilingual-e5-small')
LO, HI = 0.02, 0.995          # only these GBDT scores are re-scored
MAXLEN = 96


def _texts(wc, side):
    n = pd.read_pickle(f'{wc}/{side}/name_full.pkl').to_numpy(dtype=object)
    a = pd.read_pickle(f'{wc}/{side}/addr_full.pkl').to_numpy(dtype=object)
    return np.array([f'{x} | {y}'[:160] for x, y in zip(n, a)], dtype=object)


class Scorer:
    """Wraps tokenizer + model. ER_CE_DUMMY=1 -> a stand-in (for CPU plumbing tests only)."""
    def __init__(self, path=None):
        self.dummy = os.environ.get('ER_CE_DUMMY') == '1'
        if self.dummy:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        src = path or MODEL_NAME
        self.tok = AutoTokenizer.from_pretrained(src)
        self.model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1).to(self.dev)

    def _enc(self, a, b):
        return self.tok(list(a), list(b), truncation=True, max_length=MAXLEN, padding=True, return_tensors='pt')

    def fit(self, A, B, y, epochs=1, bs=256, lr=3e-5, log_every=2000):
        if self.dummy:
            return self
        torch = self.torch
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * int(np.ceil(len(y) / bs)); warm = max(1, steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
        lossf = torch.nn.BCEWithLogitsLoss()
        self.model.train(); step = 0; t = time.time()
        for ep in range(epochs):
            perm = np.random.default_rng(ep).permutation(len(y))
            for s in range(0, len(y), bs):
                idx = perm[s:s + bs]
                enc = {k: v.to(self.dev) for k, v in self._enc(A[idx], B[idx]).items()}
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.dev == 'cuda'):
                    logit = self.model(**enc).logits.squeeze(-1)
                loss = lossf(logit.float(), torch.tensor(y[idx], dtype=torch.float32, device=self.dev))
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % log_every == 0:
                    log(f'    CE step {step}/{steps} loss {loss.item():.4f} ({time.time() - t:.0f}s)')
        self.model.eval()
        return self

    def predict(self, A, B, bs=1024):
        if self.dummy:     # plumbing test: a noisy copy of a string-overlap score
            return np.array([len(set(a.split()) & set(b.split())) / max(1, len(set(a.split()) | set(b.split())))
                             for a, b in zip(A, B)], np.float32)
        torch = self.torch
        out = np.empty(len(A), np.float32)
        with torch.no_grad():
            for s in range(0, len(A), bs):
                enc = {k: v.to(self.dev) for k, v in self._enc(A[s:s + bs], B[s:s + bs]).items()}
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.dev == 'cuda'):
                    out[s:s + bs] = torch.sigmoid(self.model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        return out

    def save(self, d):
        if not self.dummy:
            os.makedirs(d, exist_ok=True); self.model.save_pretrained(d); self.tok.save_pretrained(d)


def _labels(cfg, wc, C):
    ids1 = np.load(f'{wc}/ids1.npy', allow_pickle=True); ids2 = np.load(f'{wc}/ids2.npy', allow_pickle=True)
    gt = read_tsv(f"{cfg['data']}/train/train_ground_truth.tsv")
    GP = pd.DataFrame([(a, b) for a, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in ms.split(',') if b],
                      columns=['s1', 'cand'])
    a = GP.s1.map(pd.Series(np.arange(len(ids1)), index=ids1)); b = GP.cand.map(pd.Series(np.arange(len(ids2)), index=ids2))
    ok = a.notna() & b.notna()
    g = a[ok].astype(np.int64).values << 32 | b[ok].astype(np.int64).values
    y = np.isin(C.i1.values.astype(np.int64) << 32 | C.i2.values.astype(np.int64), g).astype(np.int8)
    val = np.array([zlib.crc32(x.encode()) % 10 >= 8 for x in ids1])[C.i1.values]
    return y, val


def stage_ce(cfg, countries_train, countries_test):
    wm = f"{cfg['work']}/models"
    if not os.path.exists(f'{wm}/final.pkl'):
        raise RuntimeError('run stage 4 first')
    models = pickle.load(open(f'{wm}/final.pkl', 'rb'))
    ce_dir = f'{wm}/ce'
    # ---------------- 1. fine-tune on ambiguous TRAIN pairs (never on validation entities) ----------
    if os.path.exists(f'{ce_dir}/done') or os.environ.get('ER_CE_DUMMY') == '1':
        sc = Scorer(ce_dir if os.path.exists(f'{ce_dir}/done') else None)
    else:
        A, B, Y = [], [], []
        for c in countries_train:
            wc = f"{cfg['work']}/train/{c}"
            if not os.path.exists(f'{wc}/pA_train.npy'):
                continue
            C = pd.read_pickle(f'{wc}/C.pkl'); pa = np.load(f'{wc}/pA_train.npy')
            y, val = _labels(cfg, wc, C)
            sel = np.flatnonzero((~val) & (pa > LO / 2))
            T1, T2 = _texts(wc, 'E1'), _texts(wc, 'E2')
            A.append(T1[C.i1.values[sel]]); B.append(T2[C.i2.values[sel]]); Y.append(y[sel])
            log('CE train rows', c, len(sel), 'positives', int(y[sel].sum()))
        A, B, Y = np.concatenate(A), np.concatenate(B), np.concatenate(Y)
        cap = int(cfg.get('ce_train_pairs', 3_000_000))
        if len(Y) > cap:
            k = np.random.default_rng(0).choice(len(Y), cap, replace=False); A, B, Y = A[k], B[k], Y[k]
        log(f'CE fine-tuning {MODEL_NAME} on {len(Y):,} pairs ({Y.mean():.1%} positive)')
        sc = Scorer().fit(A, B, Y.astype(np.float32))
        sc.save(ce_dir); open(f'{ce_dir}/done', 'w').close()
        del A, B, Y; gc.collect()
    # ---------------- 2. tune the blend on validation -------------------------------------------------
    V = pickle.load(open(f'{wm}/val.pkl', 'rb'))
    R = V['rows']; hard = np.flatnonzero((R.p.values >= LO) & (R.p.values <= HI))
    pce = R.p.values.copy()
    for part, wc in enumerate(V['parts']):
        m = hard[R.part.values[hard] == part]
        if len(m):
            T1, T2 = _texts(wc, 'E1'), _texts(wc, 'E2')
            pce[m] = sc.predict(T1[R.l1.values[m]], T2[R.l2.values[m]])
    log(f'CE scored {len(hard):,} ambiguous validation pairs')
    best = (-1, None)
    for w in (1.0, 0.75, 0.5, 0.25, 0.0):
        D = R[['i1', 'i2']].copy(); D['p'] = w * R.p.values + (1 - w) * pce
        X = exclusive(D)
        for t in np.arange(0.40, 0.91, 0.05):
            f = macro_f05(to_sets(by_threshold(X, t), V['ids1'], V['ids2']), V['truth'])
            if f > best[0]:
                best = (f, ('threshold', w, float(t)))
        f = macro_f05(to_sets(by_expected_f(X), V['ids1'], V['ids2']), V['truth'])
        if f > best[0]:
            best = (f, ('expected_f', w, None))
        log(f'  blend w={w:.2f}: best so far F0.5 {best[0]:.4f} {best[1]}')
    rule, w, t = best[1]
    md = models['dec']
    base_f = md['f_expected'] if md['rule'] == 'expected_f' else md['f_threshold']
    dec = {'rule': rule, 'w': w, 'threshold': t, 'f_val': best[0], 'f_val_gbdt_only': base_f, 'lo': LO, 'hi': HI,
           'model': MODEL_NAME}
    json.dump(dec, open(f"{cfg['out']}/validation_ce.json", 'w'), indent=1)
    log('CE VALIDATION macro F0.5 %.4f (GBDT only %.4f) -> w=%.2f %s' % (best[0], dec['f_val_gbdt_only'], w, rule))
    # ---------------- 3. apply to TEST ----------------------------------------------------------------
    match = {}
    for c in countries_test:
        wc = f"{cfg['work']}/test/{c}"
        if not os.path.exists(f'{wc}/pred.pkl'):
            continue
        T = pd.read_pickle(f'{wc}/pred.pkl')
        p = T.p.values.copy()
        if w < 1.0:
            h = np.flatnonzero((p >= LO) & (p <= HI))
            f_ce = f'{wc}/pce.npy'
            if os.path.exists(f_ce):
                pc = np.load(f_ce)
            else:
                T1, T2 = _texts(wc, 'E1'), _texts(wc, 'E2')
                pc = sc.predict(T1[T.i1.values[h]], T2[T.i2.values[h]]); np.save(f_ce, pc)
            p[h] = w * p[h] + (1 - w) * pc
            log('test', c, f'CE re-scored {len(h):,} ambiguous pairs')
        D = T[['i1', 'i2']].copy(); D['p'] = p
        D = by_expected_f(exclusive(D)) if rule == 'expected_f' else by_threshold(exclusive(D), t)
        ids1 = np.load(f'{wc}/ids1.npy', allow_pickle=True); ids2 = np.load(f'{wc}/ids2.npy', allow_pickle=True)
        g = pd.DataFrame({'s1': ids1[D.i1.values], 'c': ids2[D.i2.values]})
        match.update(g.groupby('s1', sort=False).c.agg(','.join).to_dict())
    order = read_tsv(f"{cfg['data']}/test/test_source1.tsv", usecols=['entity_id']).entity_id.to_numpy(dtype=object)
    out = pd.DataFrame({'source1_entity_id': order})
    out['matched_entity_ids'] = out.source1_entity_id.map(match).fillna('')
    od = cfg['out'] + '_ce'; os.makedirs(od, exist_ok=True)
    out.to_csv(f'{od}/matching_results.tsv', sep='\t', index=False)
    import shutil
    shutil.copy(f"{cfg['out']}/candidate_pairs.tsv", f'{od}/candidate_pairs.tsv')   # same candidate set
    log('wrote', f'{od}/matching_results.tsv', f'{len(out):,} rows')
    return dec
