"""Stage 1: stream each source TSV in 400k-row chunks, normalise, store one column store per country/side."""
import os, gc, json, glob, zlib
import numpy as np
import pandas as pd

from normalize import normalize_name, normalize_address, street_and_locality
from utils import log, pmap, read_tsv, ColStore

KEEP = ['entity_id', 'country', 'src', 'name_full', 'name_core', 'name_sorted', 'name_phon',
        'name_concat', 'domain', 'is_domain', 'name_native', 'legal', 'addr_full', 'addr_words',
        'addr_nums', 'house_no', 'addr_empty', 'addr_native', 'street', 'locality',
        'name_distinct', 'house_dig', 'num_tok', 'first_tok', 'n_len']
INT_COLS = ['is_domain', 'name_native', 'addr_empty', 'addr_native', 'n_len']


def _digits(s):
    return ''.join(ch for ch in s if ch.isdigit())


def enrich(df):
    rows = []
    for eid, nm, ad, c in zip(df.entity_id.values, df.business_name.values,
                              df.business_address.values, df.country.values):
        n = normalize_name(nm); a = normalize_address(ad)
        st, lo = street_and_locality(a.pop('addr_parts'))
        bad = set(st.split()) | set(lo.split())
        core = n['name_core'].split()
        rows.append({'entity_id': eid, 'country': c, 'src': eid[:2], **n, **a, 'street': st, 'locality': lo,
                     'name_distinct': ' '.join(t for t in core if t not in bad),
                     'house_dig': _digits(a['house_no']),
                     'num_tok': a['addr_nums'].split()[0] if a['addr_nums'] else '',
                     'first_tok': core[0] if core else '', 'n_len': len(core)})
    if not rows:
        return pd.DataFrame({c: pd.Series([], dtype=object) for c in KEEP})
    out = pd.DataFrame(rows)[KEEP]
    for c in INT_COLS:
        out[c] = out[c].astype(np.int8)
    return out


def _count_rows(path):
    n, last = 0, b'\n'
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            n += b.count(b'\n'); last = b[-1:]
    return n - 1 + (last != b'\n')        # a final line without newline still counts


def _data_fp(cfg, split):
    d = f"{cfg['data']}/{split}"
    return {f: [os.path.getsize(f'{d}/{f}'), int(os.path.getmtime(f'{d}/{f}'))]
            for f in sorted(os.listdir(d)) if f.endswith('.tsv')}


def stage_normalise(cfg, split):
    """Streams each source file in 400k-row chunks; every chunk is a checkpoint.
    Guards: (1) if the data files changed since the cache was built (size/mtime), the split's
    cache AND the trained models are rebuilt; (2) rows processed must equal rows in the file."""
    w = f"{cfg['work']}/{split}"
    fp_now = _data_fp(cfg, split)
    fp_file = f'{w}/data_fp.json'
    if os.path.isdir(w) and (not os.path.exists(fp_file) or json.load(open(fp_file)) != fp_now):
        import shutil
        log(split, 'WARNING: cache was built from different/older data files -> deleting it and the models')
        shutil.rmtree(w, ignore_errors=True); shutil.rmtree(f"{cfg['work']}/models", ignore_errors=True)
    os.makedirs(f'{w}/_shards', exist_ok=True)
    json.dump(fp_now, open(fp_file, 'w'))
    if os.path.exists(f'{w}/countries.json'):
        log(split, 'normalise: cached'); return json.load(open(f'{w}/countries.json'))
    d = f"{cfg['data']}/{split}"
    for src in (1, 2, 3):
        path = f'{d}/{split}_source{src}.tsv'
        for ci, ch in enumerate(read_tsv(path, chunksize=400_000)):
            tag = f'{w}/_shards/s{src}_{ci:04d}'
            if os.path.exists(tag + '.done'):
                continue
            if split == 'train' and src == 1 and cfg['train_frac'] < 1:
                keep = np.array([zlib.crc32(x.encode()) % 1000 < cfg['train_frac'] * 1000 for x in ch.entity_id])
                ch = ch[keep]
            if src == 1:
                np.save(tag + '_ids.npy', ch.entity_id.to_numpy(dtype=object), allow_pickle=True)
            parts = [ch.iloc[i:i + 50_000] for i in range(0, len(ch), 50_000)]
            e = pd.concat(pmap(enrich, parts), ignore_index=True) if parts else enrich(ch)
            for c, g in e.groupby('country', sort=False):
                g.reset_index(drop=True).to_pickle(f'{tag}_{c}.pkl')
            open(tag + '.done', 'w').write(str(len(ch)) if not (split == 'train' and src == 1 and cfg['train_frac'] < 1) else '-1')
            log(split, f'source{src} chunk {ci} normalised ({len(ch):,} rows)')
            del e, ch; gc.collect()
    # every row of every file must have been processed (catches truncated / half-extracted files)
    for src in (1, 2, 3):
        got = [open(f).read() for f in glob.glob(f'{w}/_shards/s{src}_*.done')]
        if any(g in ('', '-1') for g in got):
            continue
        want = _count_rows(f'{d}/{split}_source{src}.tsv')
        if sum(int(g) for g in got) != want:
            raise RuntimeError(f'{split}_source{src}: processed {sum(int(g) for g in got):,} rows but the file has '
                               f'{want:,}. Delete {w} and re-run (the data files were probably re-extracted).')
    # consolidate per country into column stores
    shard_files = sorted(glob.glob(f'{w}/_shards/s*_*_*.pkl'))
    countries = sorted({os.path.basename(f).split('_', 2)[2][:-4] for f in shard_files})
    for c in countries:
        for side, srcs in (('E1', ('s1',)), ('E2', ('s2', 's3'))):
            dd = f'{w}/{c}/{side}'
            if ColStore.exists(dd):
                continue
            fs = [f for s_ in srcs for f in sorted(glob.glob(f'{w}/_shards/{s_}_*_{c}.pkl'))]
            df = pd.concat([pd.read_pickle(f) for f in fs], ignore_index=True) if fs else enrich(pd.DataFrame(
                columns=['entity_id', 'business_name', 'business_address', 'country']))
            ColStore.save(dd, df)
            np.save(f'{w}/{c}/ids{side[1]}.npy', df.entity_id.to_numpy(dtype=object), allow_pickle=True)
            del df; gc.collect()
        log(split, c, 'column store ready:', ColStore.n(f'{w}/{c}/E1'), 'S1 /', ColStore.n(f'{w}/{c}/E2'), 'S2+S3')
    ids = np.concatenate([np.load(f, allow_pickle=True) for f in sorted(glob.glob(f'{w}/_shards/s1_*_ids.npy'))])
    np.save(f'{w}/s1_order.npy', ids, allow_pickle=True)
    json.dump(countries, open(f'{w}/countries.json', 'w'))
    log(split, 'normalise done', countries)
    return countries
