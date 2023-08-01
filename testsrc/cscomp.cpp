#include "rnla.h"

double fracdiff(double x,  double y) {
    return std::abs(x - y) / std::max(x, y);
}

int main() {
    size_t D = 10000;
    sketch::KWiseHasherSet<4> hf(1337);
    std::vector<double> init(D, 1.);
    blaze::DynamicVector<double> init_1(D, 1.);
    for(size_t i = 0; i < init.size(); ++i) init[i] = std::pow(i, 2);
    auto step1 = cs_compress(init, 100, hf);
    auto step1_1= cs_compress(init_1, 100, hf);
    auto step1_2 = cs_compress(init, 13, hf);
    auto step2 = cs_decompress(step1, D, hf);
    auto step2_1 = cs_decompress(step1_1, D, hf);
    // top_indices_from_compressed(const C &in, size_t newdim, size_t olddim, const KWiseHasherSet<4> &hf, unsigned k)
    auto topind = top_indices_from_compressed(step1, D, 100, hf, 20);
    auto topind2 = top_indices_from_compressed(step1_2, D, 13, hf, 20);
    sketch::SketchApplicator<> sa(100, 10);
    sketch::IndykSketcher<double> is(9, 50, D, 1377);
    //size_t ntables, size_t destdim, uint64_t sourcedim=0,
    auto n = is.norm(init_1);
    std::fprintf(stderr, "Indyk-sketched norm: %g\n", n);
    wy::WyHash<uint64_t, 8> gen;
    double reall1sum = 0.;
    std::normal_distribution<double> dist;
    for(size_t i = 0; i < D; ++i) {
        for(auto &i: init_1)
            i = std::pow(dist(gen), 4);
        reall1sum += l1Norm(init_1);
        is.add(trans(init_1));
    }
    assert(std::abs(is.pnorm() - reall1sum) / reall1sum <= 1.); // Within a factor of 2
    sketch::IndykSketcher<double> is2(7, 100, 0), is3(7, 100, 0);
    size_t sn = 20000;
    double nl = 1, nr = 20;
    for(size_t i = 0; i < sn; ++i) {
        is2.addh(i, nl), is3.addh(i, -nr);
        std::swap(nl, nr);
    }
    auto us = is2.union_size(is3);
    std::fprintf(stderr, "us: %f. n1 %f, n2 %f. expected: %zu. %% diff: %f\n", us, is2.pnorm(), is3.pnorm(), sn * 19, fracdiff(us, is2.pnorm() + is3.pnorm()) * 100.);
    auto diff = is2 - is3;
}
