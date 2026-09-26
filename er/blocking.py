"""
Candidate generation (blocking) primitives.

Several cheap, complementary "views" each propose top-K Source-2/3 neighbours for every
Source-1 record, *within the same country string* (open set: whatever labels appear).
Each view = hashed char n-gram TF-IDF -> TruncatedSVD (dense, d dims) -> L2-normalised,
so the inner product approximates TF-IDF cosine similarity; top-K via FAISS / torch / numpy.
A rare-key inverted index (record_keys / key_candidates_np) complements the dense views.
"""
import os, re, time
import numpy as np
import pandas as pd

from utils import log, n_gpus, N_JOBS
from text_vectors import CharTfidf

VIEWS = {
    # name only: core + phonetic skeleton + domain/concat (catches native script & URLs)
    'name': lambda d: (d.name_core + ' | ' + d.name_phon + ' | ' + d.domain),
    # address only: catches "different trade name, same premises"
    'addr': lambda d: d.addr_full,
    # joint view
    'both': lambda d: (d.name_core + ' | ' + d.name_phon + ' | ' + d.addr_words + ' ' + d.addr_nums),
}


def topk_dense(ea, eb, k, chunk=512):
    """Exact inner-product top-k of each row of ea against eb. Returns (idx, sim)."""
    k = min(k, eb.shape[0])
    I = np.empty((ea.shape[0], k), dtype=np.int64)
    S = np.empty((ea.shape[0], k), dtype=np.float32)
    for i in range(0, ea.shape[0], chunk):
        sims = ea[i:i + chunk] @ eb.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(sims, part, axis=1)
        o = np.argsort(-ps, axis=1)
        I[i:i + chunk] = np.take_along_axis(part, o, axis=1)
        S[i:i + chunk] = np.take_along_axis(ps, o, axis=1)
    return I, S


# ---------------------------------------------------------------------------------------
# Rare-key pass: exact-key inverted index (rare tokens).  Complements the dense passes, which
# blur rare identifiers such as "14/2995", "xii/228-d" or a 3-letter brand like "pfx".
# ---------------------------------------------------------------------------------------

_DIG = re.compile(r'\d+')


def record_keys(d: pd.DataFrame):
    """List of blocking keys per record (country-prefixed)."""
    keys = []
    locs = d.locality.values if 'locality' in d.columns else None
    streets = d.street.values if ('street' in d.columns and locs is not None) else None
    for ri, (c, full, core, phon, dom, nums) in enumerate(zip(d.country.values, d.addr_full.values, d.name_core.values,
                                              d.name_phon.values, d.domain.values, d.addr_nums.values)):
        ks = set()
        at = full.split()
        for i, t in enumerate(at):
            if any(ch.isdigit() for ch in t):
                if len(t) >= 4 or '/' in t:
                    ks.add('A:' + t)
                nxt = next((w for w in at[i + 1:i + 3] if w.isalpha() and len(w) > 2), None)
                if nxt:
                    ks.add('H:' + t + '_' + nxt[:5])          # house no + street word
        for n in nums.split():
            if '/' in n or len(n) >= 4:
                ks.add('A:' + n.replace('/', ''))
        ct = core.split()
        for t in ct:
            if len(t) >= 3:
                ks.add('N:' + t)
        for a, b in zip(ct, ct[1:]):
            ks.add('B:' + a + '_' + b)
        for t in phon.split():
            if len(t) >= 3:
                ks.add('P:' + t)
        if dom:
            ks.add('D:' + dom)
        if locs is not None:            # name token scoped to the city: rare locally even if common nationally
            lw = locs[ri].split()[:2]
            for l in lw:
                for t in ct:
                    if len(t) >= 3:
                        ks.add('L:' + t + '@' + l)
                for t in phon.split():
                    if len(t) >= 3:
                        ks.add('Q:' + t + '@' + l)
        if streets is not None:         # v6: digit groups (zero-stripped), numbers / street words scoped to a city word
            g = [x.lstrip('0') or '0' for x in _DIG.findall(full)]
            for a, b in zip(g, g[1:]):
                ks.add('G:' + a + '_' + b)
            lw = list(dict.fromkeys(locs[ri].split()))[-6:]
            sw = [w for w in streets[ri].split() if len(w) >= 4]
            for l in lw:
                for x in g:
                    if len(x) >= 2:
                        ks.add('g:' + x + '@' + l)
                for w in sw:
                    if w != l:
                        ks.add('W:' + w + '@' + l)
        keys.append([c + '|' + k for k in ks])
    return keys


def _hash_keys(keys_per_record):
    """list[list[str]] -> (row_idx int32[], key int64[])"""
    rows, ks = [], []
    for r, kl in enumerate(keys_per_record):
        for k in kl:
            rows.append(r); ks.append(hash(k))
    return np.asarray(rows, dtype=np.int32), np.asarray(ks, dtype=np.int64)


def _hash_keys_chunked(d, chunk=300_000):
    """(row, key-hash) arrays built 300k records at a time (a full key list of 6M records
    would be ~15 GB of Python strings)."""
    R, H = [], []
    for s in range(0, len(d), chunk):
        r, h = _hash_keys(record_keys(d.iloc[s:s + chunk])); R.append(r + s); H.append(h)
    return (np.concatenate(R), np.concatenate(H)) if R else (np.zeros(0, np.int32), np.zeros(0, np.int64))


def key_candidates_np(E1, E2, k=20, max_df=40, chunk=200_000):
    """Same scoring as key_candidates() but memory-lean: no string joins."""
    r2, h2 = _hash_keys_chunked(E2)
    o = np.argsort(h2, kind='stable'); r2, h2 = r2[o], h2[o]
    uk, start, cnt = np.unique(h2, return_index=True, return_counts=True)
    w_key = np.log1p(len(E2) / cnt).astype(np.float32)
    keep = cnt <= max_df
    uk, start, cnt, w_key = uk[keep], start[keep], cnt[keep], w_key[keep]
    out = []
    for s in range(0, len(E1), chunk):
        r1, h1 = _hash_keys(record_keys(E1.iloc[s:s + chunk])); r1 += s
        pos = np.searchsorted(uk, h1)
        pos[pos >= len(uk)] = 0
        hit = uk[pos] == h1
        r1, pos = r1[hit], pos[hit]
        n = cnt[pos]
        rep_r1 = np.repeat(r1, n)
        rep_w = np.repeat(w_key[pos], n)
        offs = np.repeat(start[pos] - np.cumsum(n) + n, n) + np.arange(n.sum())
        rep_r2 = r2[offs]
        code = rep_r1.astype(np.int64) << 32 | rep_r2.astype(np.int64)
        uc, inv = np.unique(code, return_inverse=True)
        score = np.bincount(inv, weights=rep_w).astype(np.float32)
        i1 = (uc >> 32).astype(np.int32); i2 = (uc & 0xFFFFFFFF).astype(np.int32)
        o = np.lexsort((-score, i1))
        i1, i2, score = i1[o], i2[o], score[o]
        first = np.r_[0, np.flatnonzero(np.diff(i1)) + 1]
        rank = np.arange(len(i1)) - np.repeat(first, np.diff(np.r_[first, len(i1)]))
        m = rank < k
        out.append(pd.DataFrame({'i1': i1[m], 'i2': i2[m], 'pass': np.int8(3),
                                 'rank': rank[m].astype(np.int16), 'sim': score[m]}))
    return pd.concat(out, ignore_index=True)

# ======================================================================= dense views: embed + top-K search


def embed(texts_fit, texts, dim, seed=0):
    """hashed char TF-IDF (pruned) -> truncated SVD -> L2-normalised float32 rows."""
    from sklearn.decomposition import TruncatedSVD
    vec = CharTfidf(prune=True, min_df=2 if len(texts_fit) > 1000 else 1).fit(texts_fit, n=200_000)
    rng = np.random.default_rng(seed)
    samp = [texts_fit[i] for i in rng.choice(len(texts_fit), min(len(texts_fit), 200_000), replace=False)]
    n_comp = min(dim, vec.dim - 1, len(samp) - 1)
    if n_comp < 1:                                           # tiny group (e.g. a handful of malformed rows)
        return [np.zeros((len(arr), 1), np.float32) for arr in texts]
    svd = TruncatedSVD(n_components=n_comp, random_state=seed, algorithm='randomized', n_iter=4)
    svd.fit(vec._one(samp))
    V = svd.components_.T.astype(np.float32)                 # (vocab, dim)
    outs = []
    use_gpu = n_gpus() > 0 and os.environ.get('ER_GPU_PROJECT', '1') == '1'
    if use_gpu:
        import torch
        Vg = torch.from_numpy(V).cuda()
    for arr in texts:
        E = np.empty((len(arr), V.shape[1]), np.float32)
        for s in range(0, len(arr), 1_000_000):
            X = vec.transform(arr[s:s + 1_000_000])
            if use_gpu:
              try:
                # embedding_bag = sparse(X) @ V as a weighted sum of V rows; mature and stable kernel
                # (torch's beta sparse-CSR matmul crashed the kernel on some CUDA/GPU combinations)
                idx = torch.from_numpy(X.indices.astype(np.int64)).cuda()
                off = torch.from_numpy(X.indptr[:-1].astype(np.int64)).cuda()
                w = torch.from_numpy(X.data.astype(np.float32)).cuda()
                Y = torch.nn.functional.embedding_bag(idx, Vg, off, mode='sum', per_sample_weights=w)
                Y = torch.nn.functional.normalize(Y, dim=1)
                E[s:s + len(Y)] = Y.cpu().numpy(); del idx, off, w, Y
                continue
              except Exception as e:
                log('    GPU projection failed, CPU fallback:', str(e)[:120]); use_gpu = False
            if True:
                Y = np.asarray(X @ V); nrm = np.linalg.norm(Y, axis=1, keepdims=True); nrm[nrm == 0] = 1
                E[s:s + len(Y)] = Y / nrm
        outs.append(E)
    return outs


_FAISS_NOTE = []


def _faiss_gpu_ok():
    """The PyPI faiss-gpu-cu12 wheel ships no kernels for Hopper (H100/H200, compute 9.x): calling it
    there aborts the whole process ('no kernel image is available'). Use FAISS-GPU only below 9.0."""
    try:
        import torch
        caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return True
    ok = all(c[0] < 9 for c in caps)
    if not ok and not _FAISS_NOTE:
        _FAISS_NOTE.append(1)
        log(f'    GPU compute capability {caps[0]} (Hopper): faiss-gpu wheel unsupported -> torch top-K on GPU')
    return ok


def search(ea, eb, k, mode='exact'):
    """Top-K inner product. FAISS on all GPUs (exact flat or IVF), else torch, else numpy."""
    k = min(k, len(eb)); t = time.time()
    try:
        if os.environ.get('ER_NO_FAISS', '0') == '1':
            raise ImportError('disabled by ER_NO_FAISS')
        import faiss
        ng = faiss.get_num_gpus()
        have_faiss = True
        if ng > 0 and not _faiss_gpu_ok():
            ng, have_faiss = 0, False                # -> torch top-K on the GPU below
    except Exception:
        ng, have_faiss = 0, False
    if mode == 'auto':
        mode = 'exact' if ng > 0 else 'ivf'
    if have_faiss and ng == 0:                     # CPU-only machine (laptop / CPU server): FAISS on all cores
        d = eb.shape[1]
        faiss.omp_set_num_threads(N_JOBS)
        if mode == 'ivf' and len(eb) > 50_000:
            nlist = int(min(16384, max(256, 4 * np.sqrt(len(eb)))))
            q = faiss.IndexFlatIP(d)
            idx = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
            idx.train(np.ascontiguousarray(eb[np.random.default_rng(0).choice(len(eb), min(len(eb), 40 * nlist), replace=False)]))
            idx.add(np.ascontiguousarray(eb)); idx.nprobe = 64
        else:
            idx = faiss.IndexFlatIP(d); idx.add(np.ascontiguousarray(eb))
        I = np.empty((len(ea), k), np.int64); S = np.empty((len(ea), k), np.float32)
        for s in range(0, len(ea), 65536):
            D, J = idx.search(np.ascontiguousarray(ea[s:s + 65536]), k)
            S[s:s + 65536], I[s:s + 65536] = D, J
        log(f'    faiss-cpu-{mode}: {len(ea):,} x {len(eb):,} in {time.time() - t:.0f}s')
        return I, S
    if ng > 0:
      try:
        d = eb.shape[1]
        if mode == 'ivf':
            nlist = int(min(16384, max(256, 4 * np.sqrt(len(eb)))))
            cpu = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
            co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.useFloat16 = True
            idx = faiss.index_cpu_to_all_gpus(cpu, co=co)
            tr = eb[np.random.default_rng(0).choice(len(eb), min(len(eb), 40 * nlist), replace=False)]
            idx.train(np.ascontiguousarray(tr)); idx.add(np.ascontiguousarray(eb))
            faiss.GpuParameterSpace().set_index_parameter(idx, 'nprobe', 64)
        else:
            co = faiss.GpuMultipleClonerOptions(); co.shard = True; co.useFloat16 = True
            idx = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(d), co=co)
            idx.add(np.ascontiguousarray(eb))
        I = np.empty((len(ea), k), np.int64); S = np.empty((len(ea), k), np.float32)
        for s in range(0, len(ea), 131072):
            D, J = idx.search(np.ascontiguousarray(ea[s:s + 131072]), k)
            S[s:s + 131072], I[s:s + 131072] = D, J
        del idx
        log(f'    faiss-{mode} x{ng}: {len(ea):,} x {len(eb):,} in {time.time() - t:.0f}s')
        return I, S
      except Exception as e:   # e.g. a faiss wheel without kernels for this GPU architecture
        log('    faiss failed -> torch top-K:', str(e)[:150]); t = time.time()
    if n_gpus() > 0:
        import torch
        nb = len(eb); BLK = 1 << 16
        nbp = -(-nb // BLK) * BLK if nb > 4 * BLK else nb
        B = torch.zeros((nbp, eb.shape[1]), dtype=torch.float16, device='cuda'); B[:nb] = torch.from_numpy(eb).to('cuda', torch.float16)
        rows = int(max(64, min(8192, 0.30 * torch.cuda.mem_get_info()[0] // (2 * nbp))))
        I = np.empty((len(ea), k), np.int64); S = np.empty((len(ea), k), np.float32)

        def plain(M):
            return torch.topk(M, k, dim=1)

        def blockwise(M):   # torch.topk on a 6M-wide row runs one CTA per row; 64k-wide blocks parallelise
            r = M.shape[0]
            v1, i1 = torch.topk(M.view(r, nbp // BLK, BLK), k, dim=2)
            v2, j = torch.topk(v1.reshape(r, -1), k, dim=1)
            return v2, (j // k) * BLK + i1.reshape(r, -1).gather(1, j)
        fn = blockwise if nbp != nb else plain
        for s in range(0, len(ea), rows):
            M = torch.from_numpy(ea[s:s + rows]).to('cuda', torch.float16) @ B.T
            if nbp != nb:
                M[:, nb:] = -60000.0
            if s == 0 and fn is blockwise:          # self-check against the plain kernel, else fall back
                try:
                    va, _ = blockwise(M[:32]); vb, _ = plain(M[:32])
                    if not torch.allclose(va.float(), vb.float(), atol=1e-3):
                        raise ValueError('mismatch')
                except Exception as e:
                    log('    blockwise top-K disabled:', str(e)[:80]); fn = plain
            v, i = fn(M); del M
            I[s:s + rows] = i.cpu().numpy(); S[s:s + rows] = v.float().cpu().numpy()
        del B; torch.cuda.empty_cache()
        log(f'    torch top-K{"(blockwise)" if fn is blockwise else ""}: {len(ea):,} x {len(eb):,} (k={k}) in {time.time() - t:.0f}s')
        return I, S
    log('    WARNING: no GPU -> numpy top-K (slow)')
    return topk_dense(ea, eb, k)

UNION_NAMES = ['name', 'addr', 'both', 'key', 'rev', 'hop']


def union_candidates(parts):
    """parts: (pass id, i1, i2, rank, sim). Pass ids 4 (reverse search, all views) and 5 (2-hop) can
    hold the same pair several times -> max sim / min rank via ufunc.at."""
    i1 = np.concatenate([p[1] for p in parts]).astype(np.int64)
    i2 = np.concatenate([p[2] for p in parts]).astype(np.int64)
    uc, inv = np.unique(i1 << 32 | i2, return_inverse=True)
    del i1, i2
    names = UNION_NAMES
    C = {'i1': (uc >> 32).astype(np.int32), 'i2': (uc & 0xFFFFFFFF).astype(np.int32)}
    sims = {n: np.full(len(uc), -1, np.float32) for n in names}
    ranks = {n: np.full(len(uc), np.inf, np.float32) for n in names}
    off = 0
    for pid, a, b, r, s in parts:
        idx = inv[off:off + len(a)]; off += len(a); n = names[pid]
        if pid >= 4:
            np.maximum.at(sims[n], idx, s.astype(np.float32)); np.minimum.at(ranks[n], idx, r.astype(np.float32))
        else:
            sims[n][idx] = np.maximum(sims[n][idx], s); ranks[n][idx] = np.minimum(ranks[n][idx], r)
    for n in names:
        ranks[n][np.isinf(ranks[n])] = -1
    for n in ['addr', 'both', 'key', 'name', 'rev', 'hop']:
        C['sim_' + n] = sims[n]
    for n in ['addr', 'both', 'key', 'name', 'rev', 'hop']:
        C['rank_' + n] = ranks[n]
    C['n_passes'] = sum((sims[n] > -1).astype(np.int8) for n in names).astype(np.int8)
    return pd.DataFrame(C)
