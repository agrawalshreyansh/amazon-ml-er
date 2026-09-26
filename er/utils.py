"""Shared helpers: logging, GPU count, crash-safe parallel map, TSV reader, column store."""
import os, gc, time, json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from concurrent.futures import TimeoutError as FuturesTimeout
import pandas as pd

T0 = time.time()
N_JOBS = max(1, min(32, os.cpu_count() or 1))   # >32 workers only adds RAM/IPC overhead


def _mem():
    try:
        rss = int([l for l in open('/proc/self/status') if l.startswith('VmRSS')][0].split()[1]) / 1e6
        av = int([l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1]) / 1e6
        return f'[rss {rss:.1f}G free {av:.0f}G]'
    except Exception:
        return ''


def log(*a):
    m, s = divmod(int(time.time() - T0), 60); h, m = divmod(m, 60)
    print(f'[{h:d}:{m:02d}:{s:02d}]{_mem()}', *a, flush=True)


def n_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0

# ============================================================================ utilities


def pmap(fn, items, timeout=None):
    """Parallel map in clean forkserver workers; dead or stuck workers -> single-process redo."""
    if N_JOBS == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    ex = ProcessPoolExecutor(max_workers=N_JOBS, mp_context=mp.get_context('forkserver'))
    try:
        return list(ex.map(fn, items, timeout=timeout or 900 + 180 * len(items)))
    except (BrokenProcessPool, FuturesTimeout, TimeoutError) as e:
        log(f'WARNING: worker pool failed ({type(e).__name__}) -> redoing in one process')
        for p in list((getattr(ex, '_processes', None) or {}).values()):
            try:
                p.kill()
            except Exception:
                pass
        gc.collect()
        return [fn(x) for x in items]
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def read_tsv(path, **kw):
    # encoding_errors='replace': one corrupt byte (e.g. a truncated copy) must not kill a 5M-row read
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, quoting=3,
                       encoding='utf-8', encoding_errors='replace', **kw)


class ColStore:
    """One pickle per column -> load only what a step needs."""
    @staticmethod
    def save(d, df):
        os.makedirs(d, exist_ok=True)
        for c in df.columns:
            df[c].reset_index(drop=True).to_pickle(f'{d}/{c}.pkl')
        json.dump({'n': len(df), 'cols': list(df.columns)}, open(f'{d}/_meta.json', 'w'))

    @staticmethod
    def load(d, cols=None):
        meta = json.load(open(f'{d}/_meta.json'))
        cols = cols or meta['cols']
        return pd.DataFrame({c: pd.read_pickle(f'{d}/{c}.pkl') for c in cols})

    @staticmethod
    def n(d):
        return json.load(open(f'{d}/_meta.json'))['n']

    @staticmethod
    def exists(d):
        return os.path.exists(f'{d}/_meta.json')
