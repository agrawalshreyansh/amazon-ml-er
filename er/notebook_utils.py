"""Helpers for the driver notebook: environment check, data download + verification, config,
the crash-safe stage runner, and submission selection / checks. Stdlib-only at import time."""
import os, sys, glob, json, shutil, signal, subprocess, zipfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================ stage 0: environment
def setup_env():
    """pip-install missing libraries, print versions / GPUs / RAM, return ROOT (work + data dir)."""
    def pip(*p): subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *p], check=False)
    for mod, pkg in [('faiss', 'faiss-gpu-cu12'), ('rapidfuzz', 'rapidfuzz'), ('numba', 'numba'), ('xgboost', 'xgboost'), ('transformers', 'transformers')]:
        try: __import__(mod)
        except ImportError: pip(pkg)
    for mod in ['faiss', 'rapidfuzz', 'numba', 'xgboost', 'torch', 'transformers']:
        try:
            m = __import__(mod); extra = ''
            if mod == 'faiss': extra = f'GPUs={m.get_num_gpus()}'
            if mod == 'torch': extra = f'GPUs={m.cuda.device_count()}'
            print(f'{mod:10s} {getattr(m, "__version__", "")} {extra}')
        except Exception as e:
            print(f'{mod:10s} MISSING ({e.__class__.__name__}) -> CPU fallback')
    mem = [l for l in open('/proc/meminfo') if l.startswith(('MemTotal', 'MemAvailable'))]
    print('CPUs', os.cpu_count(), '|', ' '.join(x.split()[1] for x in mem), 'kB')
    root = '/teamspace/studios/this_studio' if os.path.isdir('/teamspace/studios/this_studio') else os.getcwd()
    print('ROOT', root, 'free %.0f GB' % (shutil.disk_usage(root).free / 1e9), '(need ~100 GB for the FULL run)')
    return root


# ============================================================================ data: download / extract / verify
DATA_URL = 'https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip'   # link from the challenge page
EXPECTED_ROWS = {'train_source1.tsv': 2206821, 'train_source2.tsv': 5034616, 'train_source3.tsv': 5285603,
                 'train_ground_truth.tsv': 2206821, 'test_source1.tsv': 1732544, 'test_source2.tsv': 4887273,
                 'test_source3.tsv': 5082316}          # data rows (header excluded), counted on the official files


def _find(root):
    hits = glob.glob(f'{root}/**/train/train_source1.tsv', recursive=True)
    return os.path.dirname(os.path.dirname(hits[0])) if hits else None


def _rows(path):
    n = 0
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''): n += block.count(b'\n')
    return n - 1


def _bad_files(data):
    bad = []
    for name, exp in EXPECTED_ROWS.items():
        f = glob.glob(f'{data}/*/{name}')
        if not f: bad.append((name, 'missing')); continue
        with open(f[0], 'rb') as fh:
            fh.seek(max(0, os.path.getsize(f[0]) - 4096)); tail = fh.read()
        try: tail[tail.find(b'\n') + 1:].decode('utf-8')
        except UnicodeDecodeError: bad.append((name, 'truncated (ends mid-character)')); continue
        n = _rows(f[0])
        if n != exp: bad.append((name, f'{n:,} rows, expected {exp:,}'))
    return bad


def prepare_data(root, s3_uri=None):
    """Download / extract student_resource.zip (or use one dropped into root) and verify every file.
    Returns the data directory (the one holding train/ and test/)."""
    z = f'{root}/student_resource.zip'
    need = 12 * 2**30
    if _find(root) is None or _bad_files(_find(root)):
        if _find(root): print('Damaged data found, re-extracting:', _bad_files(_find(root)))
        if shutil.disk_usage(root).free < need: raise SystemExit('Not enough free disk to extract (need ~12 GB): free some space first')
        if not os.path.exists(z) or not zipfile.is_zipfile(z):
            if s3_uri: subprocess.run(['aws', 's3', 'cp', s3_uri, z], check=True)
            else: urllib.request.urlretrieve(DATA_URL, z + '.part'); os.replace(z + '.part', z)
        with zipfile.ZipFile(z) as zf:
            bad_member = zf.testzip()
            if bad_member: raise SystemExit(f'zip is corrupt at {bad_member}: delete {z} and re-run to download again')
            zf.extractall(f'{root}/data')
    data = _find(root)
    problems = _bad_files(data)
    if problems: raise SystemExit(f'Data files still damaged: {problems}. Delete {root}/data and {z}, then re-run.')
    print('DATA =', data, '| all 7 files verified (row counts + clean UTF-8 ending) | free disk %.0f GB' % (shutil.disk_usage(root).free / 1e9))
    return data


# ============================================================================ config (auto-tuned to the machine)
def make_cfg(root, data, fast=False, search='exact', reset=False):
    ram_gb = int([l for l in open('/proc/meminfo') if l.startswith('MemTotal')][0].split()[1]) / 1e6
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        gpus = []
    print('GPUs:', gpus, '| RAM %.0f GB' % ram_gb, '| CPUs', os.cpu_count())
    cfg = dict(data=data, work=f"{root}/er_work_{'fast' if fast else 'full'}", out=f'{root}/output',
               train_frac=0.35 if fast else 1.0, dim=192, search=search,
               ks={'name': 30, 'addr': 20, 'both': 30}, key_k=30, rev_k=3, hop_k=3, ce_train_pairs=3_000_000,
               max_train_pairs=int(min(80e6, max(20e6, ram_gb * 0.25e6))))   # ~264 B per row, keep < ~25% of RAM
    if reset: shutil.rmtree(cfg['work'], ignore_errors=True)
    os.makedirs(cfg['work'], exist_ok=True); os.makedirs(cfg['out'], exist_ok=True)
    json.dump(cfg, open(f"{cfg['work']}/cfg.json", 'w'))
    print(cfg)
    return cfg


# ============================================================================ stage runner
def run_stage(cfg, name):
    """Run one stage in its own process. A native crash or an out-of-memory kill then ends only that
    process: the kernel survives, the exit reason is printed, and re-running resumes from checkpoints."""
    cfg_json, log_path = f"{cfg['work']}/cfg.json", f"{cfg['work']}/pipeline.log"
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    p = subprocess.Popen([sys.executable, '-X', 'faulthandler', f'{HERE}/run_stage.py', cfg_json, name],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    with open(log_path, 'a') as lf:
        for line in p.stdout:
            if 'Warning' in line or 'warnings.warn' in line: continue
            print(line, end=''); lf.write(line)
    rc = p.wait()
    if rc == 0:
        print(f'== {name}: OK'); return
    why = {-signal.SIGKILL: 'KILLED by the OS -> out of RAM. Look at the last [rss ... free ...] values above.',
           -signal.SIGSEGV: 'SEGFAULT in a native library (see the faulthandler traceback above). '
                            'If it is in torch/embedding: set ER_GPU_PROJECT=0; in faiss: set ER_NO_FAISS=1.',
           -signal.SIGABRT: 'ABORT in a native library (CUDA/FAISS). Try ER_NO_FAISS=1 or ER_GPU_PROJECT=0.'}.get(
           rc, f'exit code {rc} (Python error above)')
    raise RuntimeError(f'stage {name} failed: {why}  -- re-run this cell to resume from checkpoints')


# ============================================================================ after training / predicting
def free_train_features(cfg):
    """Train features are not needed once the models exist."""
    if os.path.exists(f"{cfg['work']}/models/final.pkl"):
        for d in glob.glob(f"{cfg['work']}/train/*/feats"): shutil.rmtree(d, ignore_errors=True)
    print('free %.0f GB' % (shutil.disk_usage(cfg['work']).free / 1e9))


def pick_best(cfg):
    """Output dir with the better VALIDATION score: GBDT only (output/) or + cross-encoder (output_ce/)."""
    v0 = json.load(open(f"{cfg['out']}/validation.json"))
    f0 = max(v0.get('f_expected', 0), v0.get('f_threshold', 0))
    f1 = json.load(open(f"{cfg['out']}/validation_ce.json"))['f_val'] if os.path.exists(f"{cfg['out']}/validation_ce.json") else 0
    best = cfg['out'] + '_ce' if f1 > f0 + 0.0005 else cfg['out']
    print(f'GBDT val {f0:.4f} | +CE val {f1:.4f} -> SUBMIT {best}/matching_results.tsv + candidate_pairs.tsv')
    d = v0.get('diagnostics')
    if d: print(json.dumps(d, indent=1)[:3000])
    return best


def check_submission(root, data, best):
    """Official validate_submission.py if it is anywhere under root, else basic format checks."""
    import pandas as pd
    v = glob.glob(f'{root}/**/validate_submission.py', recursive=True)
    if v:
        subprocess.run(['python3', v[0], '--matching', f'{best}/matching_results.tsv',
                        '--candidate', f'{best}/candidate_pairs.tsv', '--test-dir', f'{data}/test'])
    else:
        m = pd.read_csv(f"{best}/matching_results.tsv", sep='\t', dtype=str, keep_default_na=False)
        s1 = pd.read_csv(f"{data}/test/test_source1.tsv", sep='\t', dtype=str, keep_default_na=False, usecols=['entity_id'])
        assert list(m.columns) == ['source1_entity_id', 'matched_entity_ids']
        assert m.source1_entity_id.is_unique and set(m.source1_entity_id) == set(s1.entity_id)
        assert all(len(x.split(',')) == len(set(x.split(','))) for x in m.matched_entity_ids if x)
        print('basic checks PASS', len(m), 'rows')
