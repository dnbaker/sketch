#!/usr/bin/env python3
"""Round-trip test for the sketch_bcqf python binding.

Builds a bcqf, inserts keys, serializes, deserializes via from_path,
and verifies counts. Skips cleanly when the module is not built (the
python extensions are optional; run from the python/ build tree).
"""
import os
import sys
import random
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))


def main():
    try:
        from sketch_bcqf import bcqf
    except ImportError:
        print("sketch_bcqf not built; skipping")
        return 0
    rng = random.Random(1234)
    keys = [rng.getrandbits(48) for _ in range(100000)]
    f = bcqf(16, 0.9)
    for k in keys:
        f.add(k)
    assert f.size() == len(set(keys)), (f.size(), len(set(keys)))
    for k in keys[:1000]:
        assert f.count(k) == 1
    path = os.path.join(tempfile.gettempdir(), "bcqf_py_roundtrip.gz")
    f.write(path)
    g = bcqf.from_path(path)
    assert g.q() == f.q() and g.size() == f.size(), (g.q(), f.q(), g.size(), f.size())
    for k in keys[:5000]:
        assert g.count(k) == 1, k
    for k in keys[:500]:
        g.dec(k)
    assert g.size() == f.size() - 500
    os.remove(path)
    print("python bcqf round-trip OK:", g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
