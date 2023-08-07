# Python demos

This directory contains demos for using sketch_ds from within Python.


## Getting started
1. First, install python bindings from the `sketch/python` diretory
2. `import sketch_ds`
3. Start coding



## A first program
An example might be:
```
import bns
import sys
import numpy as np
from sketch_ds import hll, util as su

nsets = 5
basen = 100000
basesize = 10000
p = 12

sets = [np.random.poisson(np.random.poisson(basen), size=(np.random.poisson(basesize),)).astype(np.uint64) for i in range(nsets)]

sketches = [hll.from_np(x, p) for x in sets]
pairwise_similarity = su.tri2full(su.jaccard_matrix(sketches))
print(pairwise_similarity)
```

which, may yield output similar to:

```
[[1.         0.7324644  0.8108363  0.70000005 0.6123147 ]
 [0.7324644  1.         0.7398335  0.55596244 0.7499867 ]
 [0.8108363  0.7398335  1.         0.69787663 0.6316901 ]
 [0.70000005 0.55596244 0.69787663 1.         0.44032815]
 [0.6123147  0.7499867  0.6316901  0.44032815 1.        ]]
```



## Contents
`shakespeare.py`: downloads, compares, and finds ngram nearest neighbors in text using pure python + the `sketch_ds` library.
`shakespeare_fast.py`: same as shakespeare.py, except uses xxhash and C++ to perform the hashing faster.

