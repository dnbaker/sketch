import unittest
from sketch import hmh
from sketch import hll
import sketch_util as su

HMHP = 10
HMHSZ = 16

class TestHLL(unittest.TestCase):
    def setUp(self):
        self.sketches = [hll.hll(HMHP) for i in range(100)]
        for i, sketch in enumerate(self.sketches):
            sketch.addh(i)
            sketch.addh(i + 1)
            sketch.addh(i + 2)

    def test_id(self):
        for sketch in self.sketches:
            self.assertEqual(hll.jaccard_index(sketch, sketch),  1.)
            self.assertTrue(abs(sketch.report() - 3.) < .1)

    def test_jm_creation(self):
        ret = su.jaccard_matrix(self.sketches)

    def test_isz_creation(self):
        ret = su.intersection_matrix(self.sketches)

    def test_ctn_creation(self):
        ret = su.containment_matrix(self.sketches)

    def test_usz_creation(self):
        ret = su.union_size_matrix(self.sketches)

    def test_sc_creation(self):
        ret = su.symmetric_containment_matrix(self.sketches)


class TestHMH(unittest.TestCase):
    def setUp(self):
        self.sketches = [hmh.hmh(HMHP, HMHSZ) for i in range(100)]
        for i, sketch in enumerate(self.sketches):
            for j in range(i, i + 50):
                sketch.addh(i + j)

    def test_id(self):
        for sketch in self.sketches:
            self.assertEqual(hmh.jaccard_index(sketch, sketch),  1.)
            self.assertTrue(abs(sketch.getcard() - 50.) < 6.)

    def test_jm_creation(self):
        ret = su.jaccard_matrix(self.sketches)

    def test_isz_creation(self):
        ret = su.intersection_matrix(self.sketches)

    def test_ctn_creation(self):
        ret = su.containment_matrix(self.sketches)

    def test_usz_creation(self):
        ret = su.union_size_matrix(self.sketches)

    def test_sc_creation(self):
        ret = su.symmetric_containment_matrix(self.sketches)



if __name__ == "__main__":
    unittest.main()
