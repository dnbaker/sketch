#include "setsketch.h"
#include "hll.h"
#include <chrono>

using namespace sketch;
struct NibbleSet8: public SetSketch<uint8_t> {
    NibbleSet8(int nreg): SetSketch<uint8_t>(nreg, 8., 1., 14) {}
};
int main(int argc, char **argv) {
    const size_t n = argc <= 1 ? 1000: std::atoi(argv[1]);
    const size_t m = argc <= 2 ? 25: std::atoi(argv[2]);
    double sb = argc <= 3 ? 1.001: std::atof(argv[3]);
    const double sa = argc <= 4 ? 30: std::atof(argv[4]);
    double hlv, bc, nc, sc, n8c;
    ByteSetS lhb(m), rhb(m);
    NibbleSetS shl(m<<1), shr(m<<1);
    NibbleSet8 nshl(m<<1), nshr(m<<1);
    ShortSetS lhn(m, sb, sa), rhn(m, sb, sa);
    hll_t h(ilog2(roundup(m));
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
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Time for %zu : %0.20g\n", n, std::chrono::duration<double, std::milli>(t2 - t).count());
    hlv = h.report();
    bc = lhb.cardinality(), nc = shl.cardinality(), sc = rhn.cardinality(), n8c = nshl.cardinality();
    std::fprintf(stderr, "h report: %0.20g, %%%f error\n", hlv, std::abs(hlv - n) / n * 100.);
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
        std::fprintf(stderr, "Matching registers for Short set: %zu/%zu. Max: %u. min: %u.\n", mat, lhn.size(), *std::max_element(lhb, lhe), *std::min_element(lhb, lhe));
        auto abmu = lhn.alpha_beta_mu(rhn, lhn.cardinality(), rhn.cardinality());
        auto a = std::get<0>(abmu);
        auto b = std::get<1>(abmu);
        auto mu = std::get<2>(abmu);
        auto isz = std::max(0., mu * (1. - a - b));
        std::fprintf(stderr, "Alpha: %g. beta: %g. mu: %g. isz: %g. JI: %g\n", a, b, mu, isz, isz / mu);
    }
    std::fprintf(stderr, "Registers for nibbles: Max: %u. min: %u.\n", *std::max_element(shl.data(), shl.data() + m), *std::min_element(shl.data(), shl.data() + m));
    std::fprintf(stderr, "Registers for smallnibbles: Max: %u. min: %u.\n", *std::max_element(nshl.data(), nshl.data() + m), *std::min_element(nshl.data(), nshl.data() + m));
    std::fprintf(stderr, "Registers for bytes: Max: %u. min: %u.\n", *std::max_element(lhb.data(), lhb.data() + m), *std::min_element(lhb.data(), lhb.data() + m));
    std::fprintf(stderr, "Registers for shorts: Max: %u. min: %u.\n", *std::max_element(rhn.data(), rhn.data() + rhn.size()), *std::min_element(rhn.data(), rhn.data() + rhn.size()));
}
