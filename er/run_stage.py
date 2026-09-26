"""Runs one pipeline stage in its own process, so a native crash or an out-of-memory kill
cannot take the Jupyter kernel down; the notebook reports the exit reason instead."""
import sys, os, json, faulthandler

if __name__ == '__main__':            # guard required: forkserver workers import this file
    faulthandler.enable()             # prints a native traceback on segfault
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    cfg = json.load(open(sys.argv[1])); stage = sys.argv[2]
    ctr = lambda: json.load(open(f"{cfg['work']}/train/countries.json"))
    cte = lambda: json.load(open(f"{cfg['work']}/test/countries.json"))
    if stage == 'normalise':
        from stage1_normalise import stage_normalise
        stage_normalise(cfg, 'train'); stage_normalise(cfg, 'test')
    elif stage == 'block':
        from stage2_block import stage_block
        stage_block(cfg, 'train', ctr()); stage_block(cfg, 'test', cte())
    elif stage == 'features_train':
        from stage3_features import stage_features
        if not os.path.exists(f"{cfg['work']}/models/final.pkl"):
            stage_features(cfg, 'train', ctr())
    elif stage == 'train':
        from stage4_train import stage_train
        stage_train(cfg, ctr())
    elif stage == 'predict':
        from stage3_features import stage_features
        from stage4_train import stage_train
        from stage5_predict import stage_predict
        stage_features(cfg, 'test', cte()); stage_predict(cfg, cte(), stage_train(cfg, ctr()))
    elif stage == 'ce':
        from stage6_ce import stage_ce
        stage_ce(cfg, ctr(), cte())
    elif stage == 'selftest_crash':
        import signal; os.kill(os.getpid(), signal.SIGKILL)
    else:
        raise SystemExit('unknown stage ' + stage)
