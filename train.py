"""Train with author architecture and data semantics using current Keras."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
from lbc_runtime import ROOT, cipher

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', choices=['smoke', 'author'], default='smoke')
    p.add_argument('--rounds', type=int)
    p.add_argument('--depth', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--samples', type=int)
    p.add_argument('--validation', type=int)
    p.add_argument('--test', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--diff', nargs=2, type=lambda v: int(v, 0))
    args = p.parse_args()
    smoke = args.preset == 'smoke'
    cfg = dict(rounds=5 if smoke else 8, depth=1 if smoke else 10,
               epochs=2 if smoke else 40, samples=4096 if smoke else 1000000,
               validation=1024 if smoke else 100000, test=2048 if smoke else 100000,
               batch_size=256 if smoke else 5000, diff=[0x2000, 0] if smoke else [0x1000, 0])
    for k in cfg:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    for k in ('rounds', 'depth', 'epochs', 'samples', 'validation', 'test', 'batch_size'):
        if cfg[k] < 1:
            p.error(k + ' must be positive')
    if any(not 0 <= d <= 65535 for d in cfg['diff']) or cfg['diff'] == [0, 0]:
        p.error('diff must be a nonzero pair of 16-bit words')
    import tensorflow as tf
    import train_nets as author
    out = ROOT / 'runs' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.preset)
    out.mkdir(parents=True)
    metadata = dict(config=cfg, preset=args.preset, smoke_only=smoke,
                    source_commit='4f7b7185c1a46e5e07665eaeb81d3028381c9143',
                    python=__import__('sys').version, tensorflow=tf.__version__,
                    data='author real_differences_data: common XOR blinding for negatives')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Output:', out, flush=True)
    print('Configuration:', cfg, flush=True)
    print('SMOKE CHECK ONLY' if smoke else 'AUTHOR NOTEBOOK CONFIGURATION (unless overridden)', flush=True)
    started = time.perf_counter()
    X, y = cipher.real_differences_data(cfg['samples'], cfg['rounds'], tuple(cfg['diff']))
    V, v = cipher.real_differences_data(cfg['validation'], cfg['rounds'], tuple(cfg['diff']))
    net = author.make_resnet(depth=cfg['depth'], reg_param=1e-5)
    net.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(out / 'best.keras'), monitor='val_loss', save_best_only=True),
                 tf.keras.callbacks.LearningRateScheduler(author.cyclic_lr(10, .002, .0001)),
                 tf.keras.callbacks.CSVLogger(str(out / 'history.csv'))]
    h = net.fit(X, y, validation_data=(V, v), epochs=cfg['epochs'], batch_size=cfg['batch_size'], callbacks=callbacks, verbose=2)
    (out / 'history.json').write_text(json.dumps(h.history, indent=2), encoding='utf-8')
    del X, y, V, v
    best = tf.keras.models.load_model(out / 'best.keras', compile=False)
    T, t = cipher.real_differences_data(cfg['test'], cfg['rounds'], tuple(cfg['diff']))
    import numpy as np
    pred = best.predict(T, batch_size=cfg['batch_size'], verbose=0).reshape(-1) >= .5
    result = dict(test_accuracy=float(np.mean(pred == t)), test_samples=len(t),
                  positive_samples=int(np.sum(t == 1)), negative_samples=int(np.sum(t == 0)),
                  true_positive_rate=float(np.mean(pred[t == 1])),
                  true_negative_rate=float(np.mean(~pred[t == 0])),
                  seconds=time.perf_counter() - started, smoke_only=smoke)
    (out / 'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (ROOT / 'runs' / 'latest.json').write_text(json.dumps({'run': str(out)}, indent=2), encoding='utf-8')
    print('Independent fresh test:', result, flush=True)
    print('Saved model:', out / 'best.keras', flush=True)

if __name__ == '__main__':
    main()
