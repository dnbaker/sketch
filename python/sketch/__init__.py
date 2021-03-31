import sketch_hll as hll
import sketch_bbmh as bbmh
import sketch_util as util
import sketch_bf as bf
import sketch_hmh as hmh
import sketch_ss as setsketch
from collections import namedtuple

SetSketchParams = namedtuple("SetSketchParams", 'a, b')

def optimal_ab(maxv, minv, *, q):
    if maxv < minv:
        minv, maxv = maxv, minv
    from numpy import exp as nexp, log as nlog
    b = nexp(nlog(maxv / minv) / q)
    return SetSketchParams(b=b, a=maxv / b)
