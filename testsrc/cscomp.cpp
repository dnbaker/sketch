#include "rnla.h"


int main() {
    sketch::KWiseHasherSet<4> hf(1337);
    std::vector<double> init(1000, 1.);
    blaze::DynamicVector<double> init_1(1000, 1.);
    for(size_t i = 0; i < init.size(); ++i) init[i] = std::pow(i, 2);
    auto step1 = cs_compress(init, 100, hf);
    auto step1_1= cs_compress(init_1, 100, hf);
    auto step1_2 = cs_compress(init, 13, hf);
    auto step2 = cs_decompress(step1, 1000, hf);
    auto step2_1 = cs_decompress(step1_1, 1000, hf);
    // top_indices_from_compressed(const C &in, size_t newdim, size_t olddim, const KWiseHasherSet<4> &hf, unsigned k) 
    auto topind = top_indices_from_compressed(step1, 1000, 100, hf, 20);
    std::fprintf(stderr, "topind sizes: %zu, %zu\n", topind.first.size(), topind.second.size());
    for(const auto i: topind.first) {
        std::fprintf(stderr, "wooo %lf\n", double(i));
    }
    auto topind2 = top_indices_from_compressed(step1_2, 1000, 13, hf, 20);
    std::fprintf(stderr, "topind2 sizes: %zu, %zu\n", topind2.first.size(), topind2.second.size());
    for(const auto i: topind2.first)
        std::fprintf(stderr, "wooo %lf\n", double(i));
    std::fprintf(stderr, "run\n");
    sketch::SketchApplicator<> sa(100, 10);
    std::fprintf(stderr, "alloc'd\n");
    sketch::IndykSketcher<double> is(5, 50, 1000, 137);
    //size_t ntables, size_t destdim, uint64_t sourcedim=0,
    auto n = is.norm(init_1);
    wy::WyHash<uint64_t, 8> gen;
    double reall1sum = 0.;
    for(size_t i = 0; i < 1000; ++i) {
        std::cauchy_distribution<double> dist;
        is.add(trans(init_1));
        reall1sum += l1Norm(init_1);
        for(auto &i: init_1)
            i = std::abs(dist(gen));
    }
    std::fprintf(stderr, "pnorm: %f. real: %f\n", is.pnorm(), reall1sum);
    assert(std::abs(is.pnorm() - reall1sum) / reall1sum * 100. <= 5.);
    auto pn = is.pnorms();
    std::fprintf(stderr, "pnorms [n:%zu]\t", is.ntables());
    for(const auto p: pn)
        std::fprintf(stderr, "%f,", p);
    std::fputc('\n', stderr);
    sketch::IndykSketcher<double> is2(7, 100, 0), is3(7, 100, 0);
    size_t sn = 20000;
    size_t nl = 1, nr = 20;
    for(size_t i = 0; i < sn; ++i) {
        is2.addh(i, nl), is3.addh(i + sn / 2, nr);
        std::swap(nl, nr);
    }
    //assert((is2.pnorm() - sn) / sn * 100. <= 12.);
    //assert((is3.pnorm() - sn) / sn * 100. <= 12.);
    auto us = is2.union_size(is3);
    std::fprintf(stderr, "us: %f. n1 %f, n2 %f\n", us, is2.pnorm(), is3.pnorm());
    std::fprintf(stderr, "True union l1: %zu. est: %f\n", sn * 21, us);
}
