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
    sim = b1.jaccard(b3)
    assert min(np.abs(cards[:2] - 1000.)) < 25.
    assert min(np.abs(cards[2] - 2000.)) < 40.
    assert (0.33333333 - sim) <= .03, f"Found similarity {sim} but expected 0.33333"

    print("cards:" + ",".join(f"{x.report()}" for x in bs))


all_tests = [test_hll, test_bbmh]

__all__ = ["all_tests", "test_hll", "test_bbmh"]
