"""Simulated last-round 16-bit subkey ranking; model rounds + one round."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import time
import numpy as np
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
from lbc_runtime import ROOT, cipher

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path)
    p.add_argument('--pairs', type=int, default=200)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--key-batch', type=int, default=256)
    p.add_argument('--seed', type=int, default=20261101)
    args = p.parse_args()
    if min(args.pairs, args.trials, args.key_batch) < 1:
        p.error('pairs, trials and key-batch must be positive')
    run = args.run or Path(json.loads((ROOT / 'runs' / 'latest.json').read_text(encoding='utf-8'))['run'])
    meta = json.loads((run / 'metadata.json').read_text(encoding='utf-8'))
    cfg = meta['config']
    import tensorflow as tf
    net = tf.keras.models.load_model(run / 'best.keras', compile=False)
    rng = np.random.default_rng(args.seed)
    nr, n = cfg['rounds'] + 1, args.pairs
    print(f'{nr}-round ciphertext -> guess last 16-bit subkey -> undo one round -> {nr-1}-round network', flush=True)
    print('Scoring: author count(score > 0.5). True key only used after scoring.', flush=True)
    if meta['smoke_only']:
        print('WARNING: smoke-trained model; ranking is a pipeline check, not an attack-performance result.', flush=True)
    out = ROOT / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    results = []
    for trial in range(args.trials):
        started = time.perf_counter()
        keys = np.repeat(rng.integers(0, 65536, (5, 1), dtype=np.uint16), n, axis=1)
        ks = cipher.expand_key(keys, nr)
        plain = tuple(rng.integers(0, 65536, n, dtype=np.uint16) for _ in range(2))
        other = tuple(x ^ d for x, d in zip(plain, cfg['diff']))
        a, b = cipher.encrypt(plain, ks), cipher.encrypt(other, ks)
        scores = np.zeros(65536, dtype=np.int32)
        for start in range(0, 65536, args.key_batch):
            end = min(start + args.key_batch, 65536)
            guesses = np.repeat(np.arange(start, end, dtype=np.uint16), n)
            aa = cipher.dec_one_round(tuple(np.tile(x, end-start) for x in a), guesses)
            bb = cipher.dec_one_round(tuple(np.tile(x, end-start) for x in b), guesses)
            X = cipher.convert_to_binary([*aa, *bb])
            z = net(X, training=False).numpy().reshape(end-start, n)
            scores[start:end] = (z > .5).sum(axis=1)
            if end % 8192 == 0 or end == 65536:
                print(f'Trial {trial+1}: {end}/65536 keys, {time.perf_counter()-started:.1f}s', flush=True)
        true = int(ks[-1][0])
        # Tie interval avoids claiming a unique rank from integer-score ties.
        rank_low = int(np.sum(scores > scores[true])) + 1
        rank_high = int(np.sum(scores >= scores[true]))
        best = np.flatnonzero(scores == scores.max())
        result = dict(trial=trial+1, true_subkey=true, rank_interval_1based=[rank_low, rank_high],
                      best_candidate_count=len(best), unique_first_correct=bool(len(best)==1 and best[0]==true),
                      seconds=time.perf_counter()-started)
        np.save(out / f'scores_{trial+1}.npy', scores)
        results.append(result)
        (out / 'results.json').write_text(json.dumps(dict(model_run=str(run), model_metadata=meta,
                pairs=n, attack_rounds=nr, seed=args.seed, results=results), indent=2), encoding='utf-8')
        print(result, flush=True)
    print('Results:', out, flush=True)
    print('This ranks one last-round subkey, not the complete 80-bit master key.', flush=True)

if __name__ == '__main__':
    main()
