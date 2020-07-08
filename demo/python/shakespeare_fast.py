import subprocess
import time
import itertools
import numpy as np
import argparse
from sketch import bf, hll, util

def parse_shakespeare():

    sp = subprocess.check_output("curl -q https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt 2>/dev/null", shell=True).decode().split("THE END")[:-1]

    IDX = [515, 515, 515, 12, 514, 515, 516, 515, 516, 515, 515, 515, 515, 515, 515, 517, 517, 516, 516, 517, 515, 515, 515, 516, 516, 516, 515, 516, 515, 515, 515, 515, 515, 516, 513, 515, 515, 515]
    plays = [x[idx:] for x, idx in zip(sp[1:], IDX)]
    plays = [x[idx:] for x, idx in zip(sp[1:], IDX)]
    names = [x.split('\n')[0] for x in plays]

    def filter_text(text):
        for x in text.strip().split():
            x = x.replace(";","").replace(".", "").replace("-","").replace(",","").strip("[]")
            if not x or x.isspace() or x.isupper():
                continue
            yield x

    def sonnet2toks(sonnettext):
        return list(itertools.chain.from_iterable(
                    map(lambda x: (_x.strip(".;,[]") for _x in x),
                    map(str.split,
                    map(str.strip,
                        sonnettext.strip().split('\n'))))))[1:]

    filtered_plays = list(map(lambda x: list(filter_text(x)), plays))
    filtered_sonnets = list(map(sonnet2toks, filter(lambda x: not not x, map(str.strip, sp[0][10502:].split("\n\n")))))
    return filtered_sonnets, filtered_plays, names


def topneighbors(pdist, k):
    ret = []
    for line in pdist:
        ret.append(sorted(enumerate(line), key=lambda x: -x[1])[1:k + 1])
    return ret

if __name__ == '__main__':
    sonnets, plays, playnames = parse_shakespeare()
    '''
    for i, name in enumerate(playnames):
        print("name %d is %s" % (i, name))
    '''
    hash_ngrams = util.hash_ngrams
    t0 = time.time()
    play_ngrams = list(map(hash_ngrams, plays))
    sonnet_ngrams = list(map(hash_ngrams, sonnets))
    t1 = time.time()
    print("Time to hash: " + str(t1 - t0))
    play_hlls = [hll.from_np(x, 12) for x in play_ngrams]
    sonnet_hlls = [hll.from_np(x, 12) for x in sonnet_ngrams]
    playdist = util.tri2full(util.jaccard_matrix(play_hlls))
    sonnetdist = util.tri2full(util.jaccard_matrix(sonnet_hlls))
    for name, x in zip(playnames, topneighbors(playdist, 5)):
        top = [i[0] for i in x]
        d = [i[1] for i in x]
        neighbors = [playnames[x] for x in top]
        dstr = ["%s/%f" % (n, d) for n, d in zip(neighbors, d)]
        print("Play %s has %s for nearest neighbors" % (name, "".join(dstr)))
    for name, x in zip(map(str, range(1, len(sonnet_ngrams))), topneighbors(sonnetdist, 5)):
        top = [i[0] for i in x]
        d = [i[1] for i in x]
        neighbors = [str(x + 1) for x in top]
        dstr = ["%s/%f" % (n, d) for n, d in zip(neighbors, d)]
        print("Sonnet %s has %s for nearest neighbors" % (name, "".join(dstr)))
