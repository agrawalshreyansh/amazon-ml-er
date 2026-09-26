"""Stage 3: pairwise features (one float16 .npy per feature, resumable per feature)."""
import os, gc, json, time
import numpy as np
import pandas as pd

from text_vectors import CharTfidf, WordTfidf, word_counts, rowdot
from utils import log, ColStore


class Pairs:
    """Pairs sorted by i2, cut into E2 row blocks -> bounded memory for E2-side matrices."""
    def __init__(self, i1, i2, n2, blk=1_500_000):
        self.i1, self.i2 = i1, i2
        self.order = np.argsort(i2, kind='stable')
        s2 = i2[self.order]
        self.blocks = []
        for s in range(0, n2, blk):
            lo, hi = np.searchsorted(s2, [s, s + blk])
            if hi > lo:
                self.blocks.append((s, min(s + blk, n2), lo, hi))


def _cos(vec, t1, t2, P):
    A = vec.transform(t1)
    out = np.empty(len(P.i1), np.float32)
    for s, e, lo, hi in P.blocks:
        B = vec.transform(t2[s:e]); idx = P.order[lo:hi]
        out[idx] = rowdot(A, B, P.i1[idx], P.i2[idx] - s)
    return out


def _overlap(t1, t2, P):
    A = word_counts(t1); nA = np.diff(A.indptr).astype(np.float32)
    inter = np.empty(len(P.i1), np.float32); n2 = np.empty(len(P.i1), np.float32)
    for s, e, lo, hi in P.blocks:
        B = word_counts(t2[s:e]); idx = P.order[lo:hi]
        inter[idx] = rowdot(A, B, P.i1[idx], P.i2[idx] - s)
        n2[idx] = np.diff(B.indptr)[P.i2[idx] - s]
    n1 = nA[P.i1]; union = n1 + n2 - inter; mn = np.minimum(n1, n2)
    jac = np.where(union > 0, inter / np.maximum(union, 1), -1).astype(np.float32)
    cont = np.where(mn > 0, inter / np.maximum(mn, 1), -1).astype(np.float32)
    return jac, cont, inter


def lev_pairs(a, b):
    try:
        from rapidfuzz.process import cpdist
        from rapidfuzz.distance import Levenshtein
        return cpdist(list(a), list(b), scorer=Levenshtein.distance, workers=-1).astype(np.float32)
    except ImportError:
        return np.array([_lev(x, y) for x, y in zip(a, b)], np.float32)


def _lev(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pair_features(wc, C, out_dir, chunk=5_000_000):
    """One float16 .npy per feature; an existing file is skipped (resume per feature)."""
    os.makedirs(out_dir, exist_ok=True)
    i1, i2 = C.i1.values.astype(np.int64), C.i2.values.astype(np.int64)
    n2 = ColStore.n(f'{wc}/E2')
    P = Pairs(i1, i2, n2)
    names = []
    have = lambda *ks: all(os.path.exists(f'{out_dir}/{k}.npy') for k in ks)
    def put(k, v):
        np.save(f'{out_dir}/{k}.npy', np.asarray(v, dtype=np.float16))
    def col(side, c):
        return pd.read_pickle(f'{wc}/{side}/{c}.pkl').tolist()
    def ocol(side, c):
        return pd.read_pickle(f'{wc}/{side}/{c}.pkl').to_numpy(dtype=object)
    def group(keys, fn):
        names.extend(keys)
        if have(*keys):
            return
        t = time.time()
        vals = fn()
        vals = vals if isinstance(vals, tuple) else (vals,)
        for k, v in zip(keys, vals):
            put(k, v)
        log(f'    {",".join(keys)} ({time.time() - t:.0f}s)')
        gc.collect()
    ch = lambda **kw: (lambda c1, c2: _cos(CharTfidf(**kw).fit(col('E1', c1) + col('E2', c2)), col('E1', c1), col('E2', c2), P))
    # --- name ---
    group(['n_char'], lambda: ch()('name_core', 'name_core'))
    group(['n_full_char'], lambda: ch()('name_full', 'name_full'))
    group(['n_phon'], lambda: ch(ns=(1, 2, 3))('name_phon', 'name_phon'))
    group(['n_word'], lambda: _cos(WordTfidf(1).fit(col('E1', 'name_core') + col('E2', 'name_core')),
                                   col('E1', 'name_core'), col('E2', 'name_core'), P))
    group(['n_sorted'], lambda: ch(ns=(3,))('name_sorted', 'name_sorted'))
    group(['n_jac', 'n_cont', 'n_inter'], lambda: _overlap(col('E1', 'name_core'), col('E2', 'name_core'), P))
    group(['p_jac', 'p_cont'], lambda: _overlap(col('E1', 'name_phon'), col('E2', 'name_phon'), P)[:2])
    def concat():
        dom, nc = col('E2', 'domain'), col('E2', 'name_concat')
        d2 = [d if d else x for d, x in zip(dom, nc)]
        v = CharTfidf(ns=(3,), wb=False).fit(col('E1', 'name_concat') + d2)
        return _cos(v, col('E1', 'name_concat'), d2, P)
    group(['n_concat'], concat)
    def dom_in():
        dom, conc = ocol('E2', 'domain'), ocol('E1', 'name_concat')
        v = np.full(len(i1), -1, np.float32); h = np.flatnonzero(dom[i2] != '')
        v[h] = [float(d in c) for d, c in zip(dom[i2[h]], conc[i1[h]])]
        return v
    group(['dom_in_name'], dom_in)
    def first_eq():
        f1, f2 = ocol('E1', 'first_tok'), ocol('E2', 'first_tok')
        return np.concatenate([(f1[i1[s:s + chunk]] == f2[i2[s:s + chunk]]) & (f1[i1[s:s + chunk]] != '')
                               for s in range(0, len(i1), chunk)])
    group(['n_first_eq'], first_eq)
    def fuzzy():
        try:
            from rapidfuzz.process import cpdist
            from rapidfuzz import fuzz
            from rapidfuzz.distance import JaroWinkler
        except ImportError:
            z = np.full(len(i1), -1, np.float32); return z, z, z
        nf1, nf2 = ocol('E1', 'name_full'), ocol('E2', 'name_full')
        nc1, nc2 = ocol('E1', 'name_core'), ocol('E2', 'name_core')
        r = {k: [] for k in 'tpj'}
        for s_ in range(0, len(i1), chunk):
            a, b = i1[s_:s_ + chunk], i2[s_:s_ + chunk]
            r['t'].append(cpdist(list(nf1[a]), list(nf2[b]), scorer=fuzz.token_set_ratio, workers=-1) / 100)
            r['p'].append(cpdist(list(nc1[a]), list(nc2[b]), scorer=fuzz.partial_ratio, workers=-1) / 100)
            r['j'].append(cpdist(list(nc1[a]), list(nc2[b]), scorer=JaroWinkler.similarity, workers=-1))
        return tuple(np.concatenate(r[k]).astype(np.float32) for k in 'tpj')
    group(['rf_token_set', 'rf_partial', 'rf_jaro'], fuzzy)
    # --- address ---
    group(['a_char'], lambda: ch()('addr_full', 'addr_full'))
    group(['a_word'], lambda: _cos(WordTfidf(2).fit(col('E1', 'addr_words') + col('E2', 'addr_words')),
                                   col('E1', 'addr_words'), col('E2', 'addr_words'), P))
    group(['a_jac', 'a_cont'], lambda: _overlap(col('E1', 'addr_words'), col('E2', 'addr_words'), P)[:2])
    group(['num_jac', 'num_cont', 'num_inter'], lambda: _overlap(col('E1', 'addr_nums'), col('E2', 'addr_nums'), P))
    def house():
        H1, H2 = ocol('E1', 'house_no'), ocol('E2', 'house_no')
        D1, D2 = ocol('E1', 'house_dig'), ocol('E2', 'house_dig')
        T1, T2 = ocol('E1', 'num_tok'), ocol('E2', 'num_tok')
        int1 = np.array([int(x[:9]) if x else -1 for x in D1], np.float64)
        int2 = np.array([int(x[:9]) if x else -1 for x in D2], np.float64)
        len1 = np.array([len(x) for x in D1], np.float32); len2 = np.array([len(x) for x in D2], np.float32)
        suf1 = np.array([x[-2:] for x in D1], object); suf2 = np.array([x[-2:] for x in D2], object)
        cols = {k: [] for k in ['house_eq', 'house_dig_eq', 'house_suffix_eq', 'house_len_diff', 'house_lev', 'house_logdiff', 'num_tok_lev']}
        for s in range(0, len(i1), chunk):
            a, b = i1[s:s + chunk], i2[s:s + chunk]
            h1, h2 = H1[a], H2[b]
            cols['house_eq'].append(np.where((h1 == '') | (h2 == ''), -1, (h1 == h2).astype(np.int8)))
            d1, d2 = D1[a], D2[b]; both = (d1 != '') & (d2 != '')
            cols['house_dig_eq'].append(np.where(both, (d1 == d2).astype(np.int8), -1))
            cols['house_suffix_eq'].append(np.where(both, (suf1[a] == suf2[b]).astype(np.int8), -1))
            cols['house_len_diff'].append(np.where(both, np.abs(len1[a] - len2[b]), -1))
            lv = np.full(len(a), -1, np.float32); w_ = np.flatnonzero(both); lv[w_] = lev_pairs(d1[w_], d2[w_])
            cols['house_lev'].append(lv)
            cols['house_logdiff'].append(np.where(both, np.log1p(np.abs(int1[a] - int2[b])), -1))
            t1, t2 = T1[a], T2[b]
            lv = np.full(len(a), -1, np.float32); w_ = np.flatnonzero((t1 != '') & (t2 != '')); lv[w_] = lev_pairs(t1[w_], t2[w_])
            cols['num_tok_lev'].append(lv)
        return tuple(np.concatenate(cols[k]) for k in cols)
    group(['house_eq', 'house_dig_eq', 'house_suffix_eq', 'house_len_diff', 'house_lev', 'house_logdiff', 'num_tok_lev'], house)
    group(['addr_empty2'], lambda: pd.read_pickle(f'{wc}/E2/addr_empty.pkl').values[i2])
    group(['street_char'], lambda: ch()('street', 'street'))
    group(['street_jac', 'street_cont'], lambda: _overlap(col('E1', 'street'), col('E2', 'street'), P)[:2])
    group(['loc_jac'], lambda: _overlap(col('E1', 'locality'), col('E2', 'locality'), P)[0])
    group(['n_distinct_char'], lambda: ch()('name_distinct', 'name_distinct'))
    group(['n_distinct_jac'], lambda: _overlap(col('E1', 'name_distinct'), col('E2', 'name_distinct'), P)[0])
    def legal():
        jac, _, li = _overlap(col('E1', 'legal'), col('E2', 'legal'), P)
        L1, L2 = ocol('E1', 'legal'), ocol('E2', 'legal')
        e1, e2 = (L1 == '')[i1], (L2 == '')[i2]
        eq = np.concatenate([np.where(e1[s:s + chunk] | e2[s:s + chunk], -1,
                                      (L1[i1[s:s + chunk]] == L2[i2[s:s + chunk]]).astype(np.int8)) for s in range(0, len(i1), chunk)])
        return jac, eq, (~e1) & (~e2) & (li == 0), e1, e2
    group(['lf_jac', 'lf_eq', 'lf_conflict', 'lf_empty1', 'lf_empty2'], legal)
    def flags():
        g = lambda side, c: pd.read_pickle(f'{wc}/{side}/{c}.pkl').values
        ln1, ln2 = g('E1', 'n_len').astype(np.float32), g('E2', 'n_len').astype(np.float32)
        return (g('E2', 'name_native')[i2], g('E2', 'addr_native')[i2], g('E2', 'is_domain')[i2],
                (ocol('E2', 'src') == 'S3')[i2], ln1[i1], ln2[i2], np.abs(ln1[i1] - ln2[i2]))
    group(['native2', 'addr_native2', 'is_domain2', 'is_s3', 'n_len1', 'n_len2', 'n_len_diff'], flags)
    bl = [c for c in ['sim_addr', 'sim_both', 'sim_key', 'sim_name', 'sim_rev', 'sim_hop', 'rank_addr', 'rank_both',
                      'rank_key', 'rank_name', 'rank_rev', 'rank_hop', 'n_passes'] if c in C.columns]
    group(bl, lambda: tuple(C[c].values for c in bl))
    ld = lambda k: np.load(f'{out_dir}/{k}.npy').astype(np.float32)
    def combo():
        nb = np.maximum.reduce([ld('n_char'), ld('n_phon'), ld('n_concat')])
        return nb, nb * np.where(ld('addr_empty2') == 1, 0.5, ld('a_char'))
    group(['name_best', 'name_x_addr'], combo)
    json.dump(names, open(f'{out_dir}/_names.json', 'w'))
    return names


def stage_features(cfg, split, countries):
    for c in countries:
        wc = f"{cfg['work']}/{split}/{c}"
        if os.path.exists(f'{wc}/feats/_names.json'):
            log(split, c, 'features: cached'); continue
        if not os.path.exists(f'{wc}/C.pkl'):
            continue
        C = pd.read_pickle(f'{wc}/C.pkl')
        if len(C) == 0:
            continue
        log(split, c, f'features for {len(C):,} pairs')
        pair_features(wc, C, f'{wc}/feats')
        log(split, c, 'features done')
        del C; gc.collect()
