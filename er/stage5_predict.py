"""Stage 5: score test pairs with the trained models and write candidate_pairs.tsv / matching_results.tsv."""
import os
import numpy as np
import pandas as pd

from decide import exclusive, by_threshold, by_expected_f
from stage4_train import Store, graph_features
from utils import log, read_tsv


def stage_predict(cfg, countries, models):
    st = Store(cfg, 'test', countries)
    A, B, dec = models['A'], models['B'], models['dec']
    cand, match = {}, {}
    for p in st.parts:
        f = f"{p['wc']}/pred.pkl"
        if os.path.exists(f):
            T = pd.read_pickle(f)
        else:
            C = p['C']; n = len(C); allr = np.arange(n); pa = np.empty(n, np.float32)
            for s in range(0, n, 4_000_000):
                x = Store.X(p, allr[s:s + 4_000_000]); pa[s:s + 4_000_000] = (A[0].predict(x) + A[1].predict(x)) / 2
            p['pA'] = pa; G = graph_features(p)
            ex = [G[k] for k in dec['gnames']] + [pa]
            pb = np.empty(n, np.float32)
            for s in range(0, n, 4_000_000):
                pb[s:s + 4_000_000] = B.predict(Store.X(p, allr[s:s + 4_000_000], ex))
            T = pd.DataFrame({'i1': C.i1.values, 'i2': C.i2.values, 'p': pb}); T.to_pickle(f)
        D = by_expected_f(exclusive(T)) if dec['rule'] == 'expected_f' else by_threshold(exclusive(T), dec['threshold'])
        for dct, R in ((cand, T), (match, D)):
            g = pd.DataFrame({'s1': p['ids1'][R.i1.values], 'c': p['ids2'][R.i2.values]})
            dct.update(g.groupby('s1', sort=False).c.agg(','.join).to_dict())
        nS1 = len(p['ids1'])
        log('test', p['country'], f'S1 {nS1:,}', 'matches/S1 %.2f' % (len(D) / max(1, nS1)),
            'empty %.1f%%' % (100 * (1 - D.i1.nunique() / max(1, nS1))))
    order = read_tsv(f"{cfg['data']}/test/test_source1.tsv", usecols=['entity_id']).entity_id.to_numpy(dtype=object)
    os.makedirs(cfg['out'], exist_ok=True)
    for fn, colname, dct in (('candidate_pairs.tsv', 'candidate_entity_ids', cand), ('matching_results.tsv', 'matched_entity_ids', match)):
        out = pd.DataFrame({'source1_entity_id': order})
        out[colname] = out.source1_entity_id.map(dct).fillna('')
        out.to_csv(f"{cfg['out']}/{fn}", sep='\t', index=False)
        log('wrote', fn, f'{len(out):,} rows')
