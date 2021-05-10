#include "setsketch.h"
#include "hll.h"
#include <chrono>

using namespace sketch;
using namespace sketch::setsketch;
struct NibbleSet8: public SetSketch<uint8_t> {
    NibbleSet8(int nreg): SetSketch<uint8_t>(nreg, 8., 1., 14) {}
};


template<typename CS>
double card(const CS &o) {
    long double ret = 0.;
    auto p = o.data();
    for(size_t i = 0; i < o.size(); ++i) {
        ret += p[i];
    }
    return o.size() / ret;
}

#ifdef USE_OPH
using SType = OPCSetSketch<double>;
#else
using SType = CSetSketch<double>;
#endif
int main(int argc, char **argv) {
    if(std::find_if(argv, argv + argc, [](auto x) {return !(std::strcmp(x, "-h") && std::strcmp(x, "--help"));}) != argv + argc) {
        std::fprintf(stderr, "Usage: ./sstest [n=1000] [m=25] [shortb=1.001] [shorta=30]\n");
        std::exit(1);
    }
    const size_t n = argc <= 1 ? 1000: std::atoi(argv[1]);
    const size_t m = argc <= 2 ? 25: std::atoi(argv[2]);
    double sb = argc <= 3 ? 1.001: std::atof(argv[3]);
    const double sa = argc <= 4 ? 30: std::atof(argv[4]);
    double hlv, bc, nc, sc, n8c;
    ByteSetS lhb(m), rhb(m);
    SType css(m), css2(m), css3(m);
    //ByteSetS lhb(m, sb, sa), rhb(m, sb, sa);
    NibbleSetS shl(m<<1), shr(m<<1);
    NibbleSet8 nshl(m<<1), nshr(m<<1);
    ShortSetS lhn(m, sb, sa), rhn(m, sb, sa);
    hll_t h(ilog2(roundup(m)));
    auto t = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < n; ++i) {
        h.addh(i);
        lhb.update(i);
        rhb.update(i);
        lhn.update(i); rhn.update(i);
        shl.update(i); shr.update(i);
        nshl.update(i); nshr.update(i);
        lhn.update(i + 0xFFFFFFFFFFFull);
        lhn.update(i + 0x1FFFFFFFFFFFull);
        rhn.update(i + 0xFFFFFFFFFFFFFFull);
        rhn.update(i + 0x1FFFFFFFFFFFFFFull);
        css.update(i); css2.update(i); css3.update(i);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Time for %zu : %0.20g\n", n, std::chrono::duration<double, std::milli>(t2 - t).count());
    hlv = h.report();
    bc = lhb.cardinality(), nc = shl.cardinality(), sc = rhn.cardinality(), n8c = nshl.cardinality();
    std::fprintf(stderr, "h report: %0.20g, %%%f error\n", hlv, std::abs(hlv - n) / n * 100.);
    std::fprintf(stderr, "css est: %0.20g. %%%f error\n", card(css), std::abs((card(css) - n) / (n) * 100.));
    std::fprintf(stderr, "css est: %0.20g\n", css.cardinality());
    assert(std::equal(lhb.data(), lhb.data() + lhb.size(), rhb.data()));
    std::fprintf(stderr, "Cardinality for nibs: %0.20g, %%%f error\n", nc, std::abs(nc - n) / n * 100.);
    std::fprintf(stderr, "Cardinality for nib8s: %0.20g, %%%f error\n", n8c, std::abs(n8c - n) / n * 100.);
    std::fprintf(stderr, "Cardinality for bytes: %0.20g, %%%f error\n", bc, std::abs(bc - n) / n * 100.);
    std::fprintf(stderr, "Cardinality for shorts: %0.20g, %%%f error\n", sc, std::abs(sc - 3 * n) / (3 * n) * 100.);
    size_t mat = 0;
    for(size_t i = 0; i < lhn.size(); ++i) {
        mat += lhn[i] == rhn[i];
    }
    {
        auto lhb =lhn.data(), lhe = lhb + lhn.size();
        std::fprintf(stderr, "Matching registers for Short set: %zu/%zu. Max: %0.10Lg. min: %0.10Lg.\n", mat, lhn.size(), (long double)*std::max_element(lhb, lhe), (long double)*std::min_element(lhb, lhe));
        auto abmu = lhn.alpha_beta_mu(rhn);
        auto a = std::get<0>(abmu);
        auto b = std::get<1>(abmu);
        auto mu = std::get<2>(abmu);
        auto isz = std::max(0., mu * (1. - a - b));
        std::fprintf(stderr, "Alpha: %g. beta: %g. mu: %g. isz: %g. JI: %0.10g\n", a, b, mu, isz, isz / mu);
        auto t = std::chrono::high_resolution_clock::now();
        double mleji = lhn.jaccard_index(rhn);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "Jaccard via MLE: %0.10g in %gns\n", mleji, std::chrono::duration<double, std::nano>(t2 - t).count());
        t = std::chrono::high_resolution_clock::now();
        mleji = lhn.jaccard_index_by_card(rhn);
        t2 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "Jaccard via MLE card: %0.10g in %gns\n", mleji, std::chrono::duration<double, std::nano>(t2 - t).count());
        t = std::chrono::high_resolution_clock::now();
        mleji = lhn.jaccard_by_ix(rhn);
        t2 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "Jaccard via inclusion-exclusion: %0.10g in %gns\n", mleji, std::chrono::duration<double, std::nano>(t2 - t).count());
    }
    std::fprintf(stderr, "CSetSketch min: %0.20Lg, max: %0.20Lg\n", (long double)css.min(), (long double)css.max());
    std::fprintf(stderr, "Registers for nibbles: Max: %u. min: %u.\n", *std::max_element(shl.data(), shl.data() + m), *std::min_element(shl.data(), shl.data() + m));
    std::fprintf(stderr, "Registers for smallnibbles: Max: %u. min: %u.\n", *std::max_element(nshl.data(), nshl.data() + m), *std::min_element(nshl.data(), nshl.data() + m));
    std::fprintf(stderr, "Registers for bytes: Max: %u. min: %u.\n", *std::max_element(lhb.data(), lhb.data() + m), *std::min_element(lhb.data(), lhb.data() + m));
    std::fprintf(stderr, "Registers for shorts: Max: %u. min: %u.\n", *std::max_element(rhn.data(), rhn.data() + rhn.size()), *std::min_element(rhn.data(), rhn.data() + rhn.size()));
}
