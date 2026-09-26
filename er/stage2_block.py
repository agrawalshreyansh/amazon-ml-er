"""Stage 2: candidate pairs per country = union of dense views (+ reverse / 2-hop search) and the rare-key pass."""
import os, gc, json, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from blocking import VIEWS, key_candidates_np, embed, search, union_candidates
from utils import log, ColStore

BLOCK_COLS = ['entity_id', 'country', 'name_core', 'name_phon', 'domain', 'addr_full', 'addr_words', 'addr_nums', 'locality',
              'street']


def stage_block(cfg, split, countries):
    for c in countries:
        wc = f"{cfg['work']}/{split}/{c}"
        bfp = {'ks': cfg['ks'], 'key_k': cfg['key_k'], 'dim': cfg['dim'], 'search': cfg.get('search', 'exact'),
               'rev_k': cfg.get('rev_k', 3), 'hop_k': cfg.get('hop_k', 3), 'v': 3}
        old = json.load(open(f'{wc}/block_fp.json')) if os.path.exists(f'{wc}/block_fp.json') else None
        if old != bfp and (os.path.exists(f'{wc}/C.pkl') or glob.glob(f'{wc}/view_*.npz')):
            import shutil
            log(split, c, 'blocking settings changed -> recomputing stage 2 and everything after it')
            for f in glob.glob(f'{wc}/view_*.npz') + glob.glob(f'{wc}/emb2.npy') + [f'{wc}/C.pkl', f'{wc}/pred.pkl', f'{wc}/pA_train.npy']:
                if os.path.exists(f): os.remove(f)
            shutil.rmtree(f'{wc}/feats', ignore_errors=True); shutil.rmtree(f"{cfg['work']}/models", ignore_errors=True)
        json.dump(bfp, open(f'{wc}/block_fp.json', 'w'))
        if os.path.exists(f'{wc}/C.pkl') and os.path.exists(f'{wc}/emb2.npy'):
            log(split, c, 'block: cached'); continue
        E1 = ColStore.load(f'{wc}/E1', BLOCK_COLS); E2 = ColStore.load(f'{wc}/E2', BLOCK_COLS)
        if len(E1) == 0 or len(E2) == 0:
            pd.DataFrame({'i1': np.array([], np.int32), 'i2': np.array([], np.int32)}).to_pickle(f'{wc}/C.pkl')
            np.save(f'{wc}/emb2.npy', np.zeros((len(E2), 1), np.float16)); continue
        log(split, repr(c), f'blocking {len(E1):,} S1 x {len(E2):,} S2/S3'
            + ('   <- tiny group: probably malformed rows with an odd country value' if len(E1) < 100 else ''))
        pool = ThreadPoolExecutor(1)        # GPU search of view v overlaps CPU embedding of view v+1
        futs = {}; prev = None
        for pid, (view, k) in enumerate(cfg['ks'].items()):
            f = f'{wc}/view_{view}.npz'
            if os.path.exists(f) and (view != 'both' or os.path.exists(f'{wc}/emb2.npy')):
                continue
            ta = VIEWS[view](E1).tolist(); tb = VIEWS[view](E2).tolist()
            rs = np.random.default_rng(1)
            fit = [ta[i] for i in rs.choice(len(ta), min(len(ta), 100_000), replace=False)] + \
                  [tb[i] for i in rs.choice(len(tb), min(len(tb), 300_000), replace=False)]
            ea, eb = embed(fit, [ta, tb], cfg['dim']); del fit
            del ta, tb
            if view == 'both':                # kept for the neighbour-support features of stage 4
                np.save(f'{wc}/emb2.npy', eb.astype(np.float16))
            log(split, c, 'view', view, 'embedded')
            def job(ea=ea, eb=eb, k=k, f=f, view=view):
                mode = cfg.get('search', 'exact'); extra = {}
                I, S = search(ea, eb, k, mode)
                rk = int(cfg.get('rev_k', 3))
                if rk > 0:                    # reverse: each S2/S3 record's nearest S1 records (one owner each)
                    Ir, Sr = search(eb, ea, rk, mode); extra.update(Ir=Ir.astype(np.int32), Sr=Sr.astype(np.float16))
                hk = int(cfg.get('hop_k', 3))
                if hk > 0 and view == 'name':   # 2-hop: S2/S3 neighbours of S2/S3 records (siblings of a match)
                    H, HS = search(eb, eb, hk + 1, mode); extra.update(H=H.astype(np.int32), HS=HS.astype(np.float16))
                np.savez(f, I=I.astype(np.int32), S=S.astype(np.float32), **extra)
                log(split, c, 'view', view, 'searched')
            if prev is not None:
                prev.result()                # keep at most 2 views' embeddings in RAM
            prev = futs[view] = pool.submit(job)
            del ea, eb; gc.collect()
        kf = f'{wc}/view_key.npz'
        if not os.path.exists(kf):
            K = key_candidates_np(E1, E2, k=cfg['key_k'], max_df=40)
            np.savez(kf, i1=K.i1.values, i2=K.i2.values, r=K['rank'].values.astype(np.float32), s=K.sim.values)
            log(split, c, 'rare-key pass done', len(K)); del K
        for v, fu in futs.items():
            fu.result()                      # re-raises errors from the search thread
        pool.shutdown()
        parts = []; n1, n2 = len(E1), len(E2)
        seeds = []; H = None
        for pid, view in enumerate(cfg['ks']):
            z = np.load(f'{wc}/view_{view}.npz'); I = z['I']; kk = I.shape[1]
            parts.append((pid, np.repeat(np.arange(n1, dtype=np.int32), kk), I.ravel(),
                          np.tile(np.arange(kk, dtype=np.float32), n1), z['S'].ravel()))
            if 'Ir' in z.files:
                Ir = z['Ir']; rk = Ir.shape[1]
                parts.append((4, Ir.ravel(), np.repeat(np.arange(n2, dtype=np.int32), rk),
                              np.tile(np.arange(rk, dtype=np.float32), n2), z['Sr'].ravel().astype(np.float32)))
            if view in ('name', 'both'):
                seeds.append(I[:, :3])
            if 'H' in z.files:
                H, HS = z['H'], z['HS'].astype(np.float32)
        z = np.load(kf); parts.append((3, z['i1'], z['i2'], z['r'], z['s']))
        if int(cfg.get('hop_k', 3)) > 0 and H is not None:
            sd = np.concatenate(seeds, 1)                               # n1 x 6 seed records
            self_ = H == np.arange(n2)[:, None]                         # drop each record's self-hit
            hk = H.shape[1] - 1
            Hn = np.where(self_.any(1, keepdims=True), np.where(self_, -1, H), H)
            HSn = np.where(Hn < 0, -1, HS)
            nb = Hn[sd.ravel()].reshape(n1, -1); ns = HSn[sd.ravel()].reshape(n1, -1)
            rr = np.tile(np.arange(H.shape[1], dtype=np.float32), sd.shape[1])
            a1 = np.repeat(np.arange(n1, dtype=np.int32), nb.shape[1]); m = nb.ravel() >= 0
            parts.append((5, a1[m], nb.ravel()[m], np.tile(rr, n1)[m], ns.ravel()[m]))
            del sd, Hn, HSn, nb, ns, a1, m
            log(split, c, f'2-hop expansion (k={hk}) added')
        C = union_candidates(parts); C.to_pickle(f'{wc}/C.pkl')
        log(split, c, 'candidates', f'{len(C):,}', 'per S1 %.1f' % (len(C) / len(E1)))
        del parts, C, E1, E2; gc.collect()
