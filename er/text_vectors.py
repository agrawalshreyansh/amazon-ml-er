"""Vectorised text vectors: hashed char n-gram / word TF-IDF and a numba row-wise sparse dot."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils import log, pmap

_P = np.uint32(16777619)            # FNV-1a 32-bit prime
_SALT = {1: 0x9E3779B9, 2: 0xC2B2AE3D, 3: 0x165667B1, 4: 0x27D4EB2F, 5: 0x94D049BB, 6: 0xBF58476D}


def _pad(texts, wb):
    if wb:  # char_wb semantics: every word padded by one space, n-grams never span words
        return [' ' + '  '.join(t.split()) + ' ' if t else '' for t in texts]
    return [' ' + t + ' ' if t else '' for t in texts]


def char_counts(texts, ns=(2, 3, 4), n_features=1 << 20, wb=True, chunk=100_000, maxlen=256):
    """CSR (float32 counts) of hashed char n-grams. Rows are processed in length-sorted
    chunks so the padded byte matrix stays tight."""
    pt = _pad(list(texts), wb)
    N = len(pt)
    lens = np.fromiter((len(t) for t in pt), np.int32, N)
    order = np.argsort(lens, kind='stable')
    mask = np.uint32(n_features - 1)
    R, Cc = [], []
    for s in range(0, N, chunk):
        idx = order[s:s + chunk]
        L = int(min(maxlen, max(1, lens[idx].max())))
        arr = np.array([pt[i][:L].encode('ascii', 'ignore') for i in idx], dtype=f'S{L}')
        B = np.frombuffer(arr.tobytes(), np.uint8).reshape(len(idx), L)
        for n in ns:
            if L < n:
                continue
            W = L - n + 1
            h = np.full((len(idx), W), _SALT[n], np.uint32)
            valid = B[:, 0:W] != 0
            for j in range(n):
                cj = B[:, j:j + W]
                h ^= cj
                h *= _P
                if j:
                    valid &= cj != 0
                if wb and 0 < j < n - 1:
                    valid &= cj != 32
            if wb and n == 2:
                valid &= ~((B[:, 0:W] == 32) & (B[:, 1:1 + W] == 32))
            r, c = np.nonzero(valid)
            R.append(idx[r]); Cc.append(h[r, c] & mask)
    r = np.concatenate(R) if R else np.zeros(0, np.int64)
    c = np.concatenate(Cc).astype(np.int32) if Cc else np.zeros(0, np.int32)
    m = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(N, n_features))
    m.sum_duplicates()
    return m


class CharTfidf:
    """Hashed char n-gram TF-IDF. prune=True keeps only buckets seen >= min_df times in the fit
    sample (compact columns, needed before SVD); prune=False keeps all 2^20 buckets."""
    def __init__(self, ns=(2, 3, 4), wb=True, nf=1 << 20, prune=False, min_df=2):
        self.ns, self.wb, self.nf, self.prune, self.min_df = ns, wb, nf, prune, min_df

    def fit(self, texts, n=400_000):
        t = list(texts)
        if len(t) > n:
            t = [t[i] for i in np.random.default_rng(0).choice(len(t), n, replace=False)]
        X = char_counts(t, self.ns, self.nf, self.wb)
        df = np.bincount(X.indices, minlength=self.nf)
        self.idf = (np.log((1 + X.shape[0]) / (1 + df)) + 1).astype(np.float32)
        if self.prune:
            keep = df >= self.min_df
            self.colmap = np.full(self.nf, -1, np.int32); self.colmap[keep] = np.arange(keep.sum())
            self.idf = self.idf[keep]; self.dim = int(keep.sum())
        return self

    def _one(self, texts):
        X = char_counts(texts, self.ns, self.nf, self.wb)
        X.data = 1 + np.log(X.data)
        if self.prune:
            c = self.colmap[X.indices]; ok = c >= 0
            rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))[ok]
            X = sp.csr_matrix((X.data[ok], (rows, c[ok])), shape=(X.shape[0], self.dim))
        X.data *= self.idf[X.indices]
        n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); n[n == 0] = 1
        X = sp.diags((1 / n).astype(np.float32)).dot(X).tocsr()
        X.sort_indices()
        return X.astype(np.float32)

    def transform(self, texts, chunk=250_000):
        t = list(texts)
        parts = [t[i:i + chunk] for i in range(0, len(t), chunk)]
        if len(parts) <= 1:
            return self._one(t)
        return sp.vstack(pmap(_CharJob(self), parts), format='csr')


class _CharJob:
    def __init__(self, v): self.v = v
    def __call__(self, t): return self.v._one(t)


def word_counts(texts, ngram=1, nf=1 << 22, binary=True):
    """Whitespace tokens (+ adjacent bigrams if ngram=2) hashed with pandas' C hash -> CSR."""
    s = pd.Series(list(texts), dtype=object)
    tok = s.str.split()
    ex = tok.explode()
    ex = ex[ex.notna() & (ex != '')]
    rows = ex.index.to_numpy()
    vals = ex.to_numpy(dtype=object)
    if ngram == 2 and len(vals) > 1:
        same = rows[1:] == rows[:-1]
        big = (pd.Series(vals[:-1][same]) + ' ' + pd.Series(vals[1:][same])).to_numpy(dtype=object)
        rows = np.concatenate([rows, rows[:-1][same]]); vals = np.concatenate([vals, big])
    if len(vals) == 0:
        return sp.csr_matrix((len(s), nf), dtype=np.float32)
    cols = (pd.util.hash_array(vals) & np.uint64(nf - 1)).astype(np.int32)
    X = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(s), nf))
    X.sum_duplicates()
    if binary:
        X.data[:] = 1
    X.sort_indices()
    return X


class WordTfidf:
    def __init__(self, ngram=1, nf=1 << 22):
        self.ngram, self.nf = ngram, nf

    def fit(self, texts, n=1_000_000):
        t = list(texts)
        if len(t) > n:
            t = [t[i] for i in np.random.default_rng(0).choice(len(t), n, replace=False)]
        X = word_counts(t, self.ngram, self.nf, binary=True)
        df = np.bincount(X.indices, minlength=self.nf)
        self.idf = (np.log((1 + X.shape[0]) / (1 + df)) + 1).astype(np.float32)
        return self

    def transform(self, texts):
        X = word_counts(texts, self.ngram, self.nf, binary=False)
        X.data = (1 + np.log(X.data)) * self.idf[X.indices]
        n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); n[n == 0] = 1
        X = sp.diags((1 / n).astype(np.float32)).dot(X).tocsr()
        X.sort_indices()
        return X.astype(np.float32)


# numba row-wise sparse dot product (parallel over pairs); scipy fallback


try:
    from numba import njit, prange

    @njit(parallel=True, fastmath=True, cache=True)
    def _rowdot_nb(ap, ai, av, bp, bi, bv, ia, ib, out):
        for k in prange(ia.shape[0]):
            p = ap[ia[k]]; pe = ap[ia[k] + 1]; q = bp[ib[k]]; qe = bp[ib[k] + 1]
            s = 0.0
            while p < pe and q < qe:
                x = ai[p]; y = bi[q]
                if x == y:
                    s += av[p] * bv[q]; p += 1; q += 1
                elif x < y:
                    p += 1
                else:
                    q += 1
            out[k] = s
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


def rowdot(A, B, ia, ib, chunk=300_000):
    out = np.empty(len(ia), np.float32)
    if HAVE_NUMBA:
        try:
            _rowdot_nb(A.indptr.astype(np.int64), A.indices, A.data, B.indptr.astype(np.int64), B.indices, B.data,
                       np.ascontiguousarray(ia, np.int64), np.ascontiguousarray(ib, np.int64), out)
            return out
        except Exception as e:
            log('numba rowdot failed, scipy fallback:', e)
    for s in range(0, len(ia), chunk):
        out[s:s + chunk] = np.asarray(A[ia[s:s + chunk]].multiply(B[ib[s:s + chunk]]).sum(axis=1)).ravel()
    return out
