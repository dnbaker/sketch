## Python demos

This directory contains demos for using sketch from within Python.

1. First, install python bindings from the `sketch/python` diretory
2. `import sketch`
3. Start coding


A first program might be:
```
import bns
import sys
import numpy as np
from sketch import hll, util as su

nsets = 5
basen = 100000
basesize = 10000
p = 12

sets = [np.random.poisson(np.random.poisson(basen), size=(np.random.poisson(basesize),)).astype(np.uint64) for i in range(nsets)]

sketches = [hll.from_np(x, p) for x in sets]
pairwise_similarity = su.tri2full(su.jaccard_matrix(sketches))
print(pairwise_similarity)

```
