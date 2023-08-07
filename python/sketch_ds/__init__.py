import sketch_hll as hll
import sketch_bbmh as bbmh
import sketch_util as util
import sketch_bf as bf
import sketch_hmh as hmh
import sketch_ss as setsketch
import sketch_lsh as lsh
from .sketch_test import all_tests
from collections import namedtuple

from sketch_util import *

SetSketchParams = namedtuple("SetSketchParams", 'a, b')


def test():
    for test in all_tests:
        test()


def optimal_ab(maxv, minv, *, q):
    '''
        Calculate a and b for maxv and minv, such that the maxv is mapped to
        0 and minv's value is mapped to q.
        :param maxv: float value which is the maximum to be quantized
        :param minv: float value which is the minimum to be quantized
        :param q:    float or integral value for the ceiling; required.
        :return: namedtuple SetSketchParams, consisting of (a, b);
                 access through ssp.a, ssp[0], or tuple access
    '''

    if maxv < minv:
        minv, maxv = maxv, minv
    from numpy import exp as nexp, log as nlog
    b = nexp(nlog(maxv / minv) / q)
    return SetSketchParams(b=b, a=maxv / b)
