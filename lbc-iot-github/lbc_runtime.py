"""Local adapter; upstream source stays unchanged."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'upstream' / 'LBC-IoT_Github'))
import lbciot as cipher

reference_substitute = cipher.substitute

def substitute(x, s):
    # Same nibble substitution, without the upstream function's input mutation.
    x = np.asarray(x, dtype=np.uint16)
    table = np.asarray(s, dtype=np.uint16)
    y = np.zeros_like(x)
    for shift in (0, 4, 8, 12):
        y |= table[(x >> shift) & 15] << shift
    return y

cipher.substitute = substitute

def check():
    x = np.arange(65536, dtype=np.uint16)
    np.testing.assert_array_equal(substitute(x, cipher.S), reference_substitute(x.copy(), cipher.S))
    rng = np.random.default_rng(20261101)
    keys = rng.integers(0, 65536, (5, 256), dtype=np.uint16)
    p = tuple(rng.integers(0, 65536, 256, dtype=np.uint16) for _ in range(2))
    ks = cipher.expand_key(keys, 8)
    fast = cipher.encrypt(p, ks)
    cipher.substitute = reference_substitute
    try:
        slow = cipher.encrypt(p, ks)
    finally:
        cipher.substitute = substitute
    for a, b in zip(fast, slow):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(p, cipher.decrypt(fast, ks)):
        np.testing.assert_array_equal(a, b)
    X, y = cipher.real_differences_data(256, 7, (0x1000, 0))
    assert X.shape == (256, 64) and y.shape == (256,)
    assert set(np.unique(X)) <= {0, 1}
    assert set(np.unique(y)) == {0, 1}
    print('[PASS] All 65536 S-box inputs match upstream; 8-round encryption matches upstream.')
    print('[PASS] Encrypt/decrypt round trip; 7-round data generation (256, 64).')
    print('These are consistency checks, not an independent official LBC-IoT test vector.')

if __name__ == '__main__':
    check()
