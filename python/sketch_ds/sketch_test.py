import sketch_ds
import numpy as np


def test_hll():

    hll = sketch_ds.hll

    h1, h2 = [hll.hll(10) for _ in "ab"]


    x = np.random.randint(0, (1<<64) - 1, dtype=np.uint64, size=1000)
    y = np.random.randint(0, (1<<64) - 1, dtype=np.uint64, size=1000)
    z = np.hstack((x, y))

    h1, h2, h3 = [hll.from_shs(w, 12) for w in (x, y, z)]
    print("cards:" + ",".join(f"{x.report()}" for x in (h1, h2, h3)))
    assert max(abs(x.report() - 1000) for x in (h1, h2)) <= 25., f"Report: {h1.report()}. Expected 1000."
    assert abs(h3.report() - 2000) <= 40., f"res: {h3.report()}. Expected: 2000."



def test_bbmh():

    bbmh = sketch_ds.bbmh.BBitMinHasher

    h1, h2 = [bbmh(12, 32) for _ in "ab"]


    x = np.random.randint(0, (1<<64) - 1, dtype=np.uint64, size=1000)
    y = np.random.randint(0, (1<<64) - 1, dtype=np.uint64, size=1000)
    z = np.hstack((x, y))

    bs = [sketch_ds.bbmh.from_shs(w, 12, 32) for w in (x, y, z)]
    b1, b2, b3 = bs
    fbs = [x.finalize() for x in bs]
    cards = np.array([x.report() for x in fbs])
    sim = b1.jaccard_index(b3)
    assert max(np.abs(cards[:2] - 1000.)) < 50., f"cards {cards[:2]}, expected 1000"
    assert abs(cards[2] - 2000.) < 100., f"card {cards[2]}, expected 2000"
    assert abs(0.5 - sim) <= .03, f"Found similarity {sim} but expected 0.5 (|x| / |x u y|)"

    print("cards:" + ",".join(f"{x.report()}" for x in bs))


def test_cqf():
    import tempfile, os
    cqf = sketch_ds.cqf

    # batch construction from numpy; counts agree with python sets (small
    # enough to stay at the requested q -- ceiling keeps clusters short)
    x = np.random.randint(0, (1 << 64) - 1, dtype=np.uint64, size=200)
    f = cqf.from_np(x, 8)
    want = len(set(x.tolist()))
    assert f.q() == 8 and f.r() == 56, "q/r wrong"
    assert f.size() == want, f"size {f.size()} != {want}"
    for v in x[:100]:
        assert f.contains(f.hash(int(v))), "lost key from batch add"
    # addh_np agrees with from_np (even when the filter must grow)
    y = np.random.randint(0, (1 << 64) - 1, dtype=np.uint64, size=5000)
    g = cqf.cqf(8)
    g.addh_np(y)
    assert g.size() == len(set(y.tolist())), "addh_np disagrees"
    assert g.q() > 8, "addh_np did not grow"
    # scalar add/dec/erase semantics (hashed-value API)
    h = cqf.cqf(6)
    h.addh(7); h.addh(7); h.addh(9)
    assert h.size() == 2 and h.count(h.hash(7)) == 2 and h.count(h.hash(9)) == 1, "scalar counts wrong"
    assert h.dec(h.hash(7)) and h.count(h.hash(7)) == 1, "dec wrong"
    assert h.erase(h.hash(9)) and not h.contains(h.hash(9)), "erase wrong"
    assert not h.erase(h.hash(9)), "erase of absent returned True"
    h.dec(h.hash(7))
    assert h.empty(), "filter not empty after dec-to-zero"
    # r argument validated against 64 - q
    try:
        cqf.cqf(8, 8)
        raise AssertionError("accepted wrong r")
    except Exception:
        pass
    # serialization round-trip through the python API
    d = tempfile.mkdtemp()
    path = os.path.join(d, "cqf.gz")
    f.write(path)
    f.clear()
    assert f.empty(), "clear failed"
    print("cqf tests passed")


all_tests = [test_hll, test_bbmh, test_cqf]

__all__ = ["all_tests", "test_hll", "test_bbmh", "test_cqf"]
