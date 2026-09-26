"""Stage 4: two-stage XGBoost (A: pair features, B: + graph features from A), threshold / rule tuned on validation."""
import os, gc, json, pickle, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from decide import exclusive, by_threshold, by_expected_f, to_sets
from metric import macro_f05
from utils import log, n_gpus, read_tsv, N_JOBS


def competition_features(i1, i2, s):
    """Graph context from a pair score, numpy sort-based (no pandas groupby)."""
    n = len(s); out = {}
    for side, ids in (('1', i1), ('2', i2)):
        o = np.lexsort((-s, ids)); ss = s[o]; gi = ids[o]
        start = np.r_[0, np.flatnonzero(np.diff(gi)) + 1]
        size = np.diff(np.r_[start, n])
        grp = np.repeat(np.arange(len(start)), size)
        rank = (np.arange(n) - start[grp] + 1).astype(np.float32)
        top = ss[start][grp]
        r = np.empty(n, np.float32)
        if side == '1':
            r[o] = rank; out['a_rank1'] = r.copy()
            r[o] = top - ss; out['a_gap1'] = r.copy()
            close = np.add.reduceat((ss > top - 0.1).astype(np.float32), start)[grp]
            r[o] = close; out['a_n_close1'] = r.copy()
            r[o] = np.add.reduceat(ss, start)[grp]; out['a_sum1'] = r.copy()
        else:
            r[o] = rank; out['a_rank2'] = r.copy()
            second = np.where(size > 1, ss[np.minimum(start + 1, n - 1)], -1.0)[grp]
            other = np.where(rank == 1, second, top)
            r[o] = ss - other; out['a_margin2'] = r.copy()
            r[o] = size[grp]; out['a_n_owner2'] = r.copy()
    return {k: out[k] for k in ['a_rank1', 'a_gap1', 'a_n_close1', 'a_sum1', 'a_rank2', 'a_margin2', 'a_n_owner2']}


def support_features(i1, i2, p, emb, M=5, chunk=2_000_000):
    """S2/S3 hold several records of the same business. For pair (a, b): take the M most likely
    candidates b' of a (by stage-A score) and measure how similar b is to them. A record that looks
    weak against the S1 name but is a near-copy of a confident match is probably a match too.
      s_max  = max_m p(a,b'_m) * cos(b, b'_m)      s_cnt = #{m : p>0.5 and cos>0.8}
      s_top  = cos(b, best b')                     (b' != b; emb = L2-normalised 'both' view)"""
    n = len(p)
    out = {'s_max': np.full(n, -1, np.float32), 's_cnt': np.zeros(n, np.float32), 's_top': np.full(n, -1, np.float32)}
    if n == 0 or emb.shape[1] < 2:
        return out
    o = np.lexsort((-p, i1)); si1 = i1[o]
    start = np.r_[0, np.flatnonzero(np.diff(si1)) + 1]; size = np.diff(np.r_[start, n])
    grp = np.repeat(np.arange(len(start)), size); pos = np.arange(n) - start[grp]
    Ai = np.full((len(start), M), -1, np.int64); Ap = np.zeros((len(start), M), np.float32)
    sel = pos < M
    Ai[grp[sel], pos[sel]] = i2[o][sel]; Ap[grp[sel], pos[sel]] = p[o][sel]
    gp = np.empty(n, np.int64); gp[o] = grp
    for s in range(0, n, chunk):
        b = i2[s:s + chunk]; g = gp[s:s + chunk]
        e = np.asarray(emb[b], np.float32)
        smax = np.full(len(b), -1, np.float32); scnt = np.zeros(len(b), np.float32); stop = np.full(len(b), -1, np.float32)
        for m in range(M):
            a = Ai[g, m]; ok = (a >= 0) & (a != b)
            cos = np.einsum('ij,ij->i', e, np.asarray(emb[np.where(ok, a, 0)], np.float32))
            cos = np.where(ok, cos, -1)
            w = np.where(ok, Ap[g, m] * cos, -1)
            smax = np.maximum(smax, w)
            scnt += ok & (Ap[g, m] > 0.5) & (cos > 0.8)
            stop = np.where((stop == -1) & ok, cos, stop)
        out['s_max'][s:s + chunk], out['s_cnt'][s:s + chunk], out['s_top'][s:s + chunk] = smax, scnt, stop
    return out


def graph_features(p_):
    """competition (one-owner) + neighbour-support features from the stage-A score p_['pA']."""
    i1, i2 = p_['C'].i1.values.astype(np.int64), p_['C'].i2.values.astype(np.int64)
    G = competition_features(i1, i2, p_['pA'])
    emb = np.load(f"{p_['wc']}/emb2.npy", mmap_mode='r') if os.path.exists(f"{p_['wc']}/emb2.npy") else np.zeros((1, 1), np.float16)
    G.update(support_features(i1, i2, p_['pA'], emb))
    return G


class Model:
    """XGBoost (GPU) with a CPU fallback. predict() returns P(match)."""
    def __init__(self, seed, gpu=None):
        self.seed, self.gpu, self.bst, self.sk = seed, gpu, None, None

    def fit(self, X, y, w=None):
        w = np.ones(len(y), np.float32) if w is None else w
        try:
            import xgboost as xgb
            dev = f'cuda:{self.gpu}' if self.gpu is not None else 'cpu'
            r = np.random.default_rng(self.seed).random(len(y)) < 0.05
            dtr = xgb.QuantileDMatrix(X[~r], y[~r], weight=w[~r], max_bin=256)
            dva = xgb.QuantileDMatrix(X[r], y[r], weight=w[r], ref=dtr)
            p = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist', device=dev,
                     max_depth=0, max_leaves=127, grow_policy='lossguide', eta=0.05, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, max_bin=256, seed=self.seed,
                     nthread=N_JOBS)
            self.bst = xgb.train(p, dtr, num_boost_round=3000, evals=[(dva, 'val')],
                                 early_stopping_rounds=50, verbose_eval=250)
            self.best = self.bst.best_iteration + 1
            return self
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.sk = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.08, max_leaf_nodes=63,
                                                     min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
                                                     validation_fraction=0.1, n_iter_no_change=30,
                                                     random_state=self.seed).fit(X, y, sample_weight=w)
            return self

    def predict(self, X, chunk=4_000_000):
        out = np.empty(len(X), np.float32)
        for s in range(0, len(X), chunk):
            x = np.asarray(X[s:s + chunk], np.float32)
            out[s:s + chunk] = (self.bst.inplace_predict(x, iteration_range=(0, self.best)) if self.bst is not None
                                else self.sk.predict_proba(x)[:, 1])
        return out


class Store:
    def __init__(self, cfg, split, countries):
        self.parts = []
        for c in countries:
            wc = f"{cfg['work']}/{split}/{c}"
            if not os.path.exists(f'{wc}/feats/_names.json'):
                continue
            names = json.load(open(f'{wc}/feats/_names.json'))
            self.parts.append(dict(country=c, wc=wc, names=names,
                                   cols=[np.load(f'{wc}/feats/{k}.npy', mmap_mode='r') for k in names],
                                   C=pd.read_pickle(f'{wc}/C.pkl')[['i1', 'i2']],
                                   ids1=np.load(f'{wc}/ids1.npy', allow_pickle=True),
                                   ids2=np.load(f'{wc}/ids2.npy', allow_pickle=True)))

    @staticmethod
    def X(p, idx, extra=None):
        cols = p['cols'] + (extra or [])
        out = np.empty((len(idx), len(cols)), np.float32)
        for j, c in enumerate(cols):
            out[:, j] = c[idx]
        return out


def stage_train(cfg, countries):
    wm = f"{cfg['work']}/models"; os.makedirs(wm, exist_ok=True)
    if os.path.exists(f'{wm}/final.pkl'):
        log('train: cached'); return pickle.load(open(f'{wm}/final.pkl', 'rb'))
    st = Store(cfg, 'train', countries)
    gt = read_tsv(f"{cfg['data']}/train/train_ground_truth.tsv")
    GP = pd.DataFrame([(a, b) for a, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for b in ms.split(',') if b],
                      columns=['s1', 'cand'])
    for p in st.parts:
        C = p['C']
        a = GP.s1.map(pd.Series(np.arange(len(p['ids1'])), index=p['ids1']))
        b = GP.cand.map(pd.Series(np.arange(len(p['ids2'])), index=p['ids2']))
        ok = a.notna() & b.notna()
        g = a[ok].astype(np.int64).values << 32 | b[ok].astype(np.int64).values
        p['y'] = np.isin(C.i1.values.astype(np.int64) << 32 | C.i2.values.astype(np.int64), g).astype(np.int8)
        bk = np.array([zlib.crc32(x.encode()) % 10 for x in p['ids1']], np.int8)[C.i1.values]
        p['val'] = bk >= 8; p['half'] = (bk >= 4).astype(np.int8)
        n_true = int(a.notna().sum())               # every true pair of this country's S1 entities
        rec = p['y'].sum() / max(1, n_true)
        log('train', repr(p['country']), f"pairs {len(C):,} positives {int(p['y'].sum()):,} of {n_true:,} true pairs"
            f" -> blocking recall {rec:.3f}")
        if n_true > 1000 and rec < 0.85:
            log('WARNING: blocking recall is far below the ~0.98 measured in development -> data or stage-2 cache '
                'is inconsistent; delete the work folder and re-run from stage 1.')
    del GP; gc.collect()
    total = sum(len(p['y']) for p in st.parts)
    # v6: keep (almost) every positive, subsample negatives, re-weight them -> calibrated probabilities
    cap = cfg['max_train_pairs']; npos = int(sum(p['y'].sum() for p in st.parts)); nneg = total - npos
    rpos = min(1.0, 0.4 * cap / max(1, npos)); rneg = min(1.0, max(0.01, (cap - rpos * npos) / max(1, nneg)))
    wneg = rpos / rneg
    log(f'training rows: positives kept {rpos:.0%}, negatives kept {rneg:.1%} (weight {wneg:.2f})')
    rng = np.random.default_rng(0)
    masks = {p['country']: rng.random(len(p['y'])) < np.where(p['y'] == 1, rpos, rneg) for p in st.parts}

    def gather(sel, extra=None):
        Xs, ys = [], []
        for p in st.parts:
            idx = np.flatnonzero(sel(p) & masks[p['country']])
            Xs.append(Store.X(p, idx, extra(p) if extra else None)); ys.append(p['y'][idx])
        y = np.concatenate(ys)
        return np.vstack(Xs), y, np.where(y == 1, 1.0, wneg).astype(np.float32)

    ng = n_gpus()
    A = [None, None]
    for h in (0, 1):
        if os.path.exists(f'{wm}/A{h}.pkl'):
            A[h] = pickle.load(open(f'{wm}/A{h}.pkl', 'rb'))

    def fitA(h):
        X, y, w = gather(lambda p: (~p['val']) & (p['half'] == h))
        log(f'stage A{h}: fitting on {len(y):,} rows' + (f' (GPU {h % ng})' if ng else ''))
        m = Model(h, gpu=(h % ng) if ng else None).fit(X, y, w)
        pickle.dump(m, open(f'{wm}/A{h}.pkl', 'wb')); log(f'stage A{h} done')
        return m
    todo = [h for h in (0, 1) if A[h] is None]
    if ng >= 2 and len(todo) == 2:
        with ThreadPoolExecutor(2) as ex:                  # one model per GPU, concurrently
            A = list(ex.map(fitA, (0, 1)))
    else:
        for h in todo:
            A[h] = fitA(h)
    for p in st.parts:
        f = f"{p['wc']}/pA_train.npy"
        if os.path.exists(f):
            p['pA'] = np.load(f)
        else:
            pa = np.empty(len(p['y']), np.float32)
            for h in (0, 1):
                o = np.flatnonzero((~p['val']) & (p['half'] != h))
                for s in range(0, len(o), 4_000_000):
                    pa[o[s:s + 4_000_000]] = A[h].predict(Store.X(p, o[s:s + 4_000_000]))
            v = np.flatnonzero(p['val'])
            for s in range(0, len(v), 4_000_000):
                x = Store.X(p, v[s:s + 4_000_000]); pa[v[s:s + 4_000_000]] = (A[0].predict(x) + A[1].predict(x)) / 2
            np.save(f, pa); p['pA'] = pa
        p['G'] = graph_features(p)
        log('stage A scores + graph features', p['country'])
    gnames = list(st.parts[0]['G'])
    extra = lambda p: [p['G'][k] for k in gnames] + [p['pA']]
    if os.path.exists(f'{wm}/B.pkl'):
        B = pickle.load(open(f'{wm}/B.pkl', 'rb'))
    else:
        X, y, w = gather(lambda p: ~p['val'], extra)
        log(f'stage B: fitting on {len(y):,} rows')
        B = Model(7, gpu=0 if ng else None).fit(X, y, w); del X, y, w; gc.collect()
        pickle.dump(B, open(f'{wm}/B.pkl', 'wb'))
    rows, o1, o2 = [], 0, 0
    for p in st.parts:
        v = np.flatnonzero(p['val'])
        pb = np.concatenate([B.predict(Store.X(p, v[s:s + 4_000_000], extra(p)))
                             for s in range(0, len(v), 4_000_000)]) if len(v) else np.zeros(0, np.float32)
        rows.append(pd.DataFrame({'i1': p['C'].i1.values[v].astype(np.int64) + o1,
                                  'i2': p['C'].i2.values[v].astype(np.int64) + o2, 'p': pb,
                                  'part': np.int16(len(rows)), 'l1': p['C'].i1.values[v], 'l2': p['C'].i2.values[v]}))
        o1 += len(p['ids1']); o2 += len(p['ids2'])
    VR = pd.concat(rows, ignore_index=True)
    Vx = exclusive(VR[['i1', 'i2', 'p']])
    ids1 = np.concatenate([p['ids1'] for p in st.parts]); ids2 = np.concatenate([p['ids2'] for p in st.parts])
    val_ids = set(x for x in ids1 if zlib.crc32(x.encode()) % 10 >= 8)
    truth = {k: set(v.split(',')) - {''} for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids) if k in val_ids}
    pickle.dump({'rows': VR, 'ids1': ids1, 'ids2': ids2, 'truth': truth, 'parts': [p['wc'] for p in st.parts]},
                open(f'{wm}/val.pkl', 'wb'))       # used by the optional cross-encoder stage
    grid = [(macro_f05(to_sets(by_threshold(Vx, t), ids1, ids2), truth), float(t)) for t in np.arange(0.40, 0.91, 0.025)]
    best_f, best_t = max(grid)
    f_exp = macro_f05(to_sets(by_expected_f(Vx), ids1, ids2), truth)
    rule = 'expected_f' if f_exp > best_f else 'threshold'
    log('VALIDATION macro F0.5: threshold %.4f (t=%.3f) | expected-F %.4f -> %s' % (best_f, best_t, f_exp, rule))
    try:   # where the points are lost (per country, error type) -> guides the next iteration
        from metric import f_beta_entity
        pred = to_sets(by_expected_f(Vx) if rule == 'expected_f' else by_threshold(Vx, best_t), ids1, ids2)
        c_of = {x: p['country'] for p in st.parts for x in p['ids1']}
        rows_ = []
        for k_, tv in truth.items():
            pv = pred.get(k_, set()); tp = len(pv & tv)
            cat = ('singleton_merged' if not tv and pv else 'fp+fn' if pv - tv and tv - pv else 'fp_only' if pv - tv
                   else 'fn_only' if tv - pv else 'perfect')
            rows_.append((c_of.get(k_, '?'), cat, f_beta_entity(pv, tv)))
        D_ = pd.DataFrame(rows_, columns=['country', 'cat', 'f'])
        diag = {'by_country': D_.groupby('country').f.mean().round(4).to_dict(),
                'points_lost_by_error': (D_.groupby('cat').f.apply(lambda x: (1 - x).sum()) / len(D_) * 100).round(3).to_dict()}
        log('DIAGNOSTICS', json.dumps(diag))
    except Exception as e:
        diag = {'error': str(e)}
    dec = {'threshold': best_t, 'f_threshold': best_f, 'f_expected': f_exp, 'rule': rule, 'grid': grid, 'gnames': gnames,
           'diagnostics': diag}
    models = {'A': A, 'B': B, 'dec': dec}
    pickle.dump(models, open(f'{wm}/final.pkl', 'wb'))
    os.makedirs(cfg['out'], exist_ok=True)
    json.dump(dec, open(f"{cfg['out']}/validation.json", 'w'), indent=1)
    return models
