#include "bbmh.h"
#include "hll.h"
#include <ostream>

using namespace sketch;
using namespace common;
using namespace mh;

#ifndef SIMPLE_HASH
#define SIMPLE_HASH 1
#endif

template<typename T>
struct scope_executor {
    T x_;
    scope_executor(T &&x): x_(std::move(x)) {}
    scope_executor(const T &x): x_(x) {}
    ~scope_executor() {x_();}
};

static bool superverbose = false;

void verify_correctness() {
    std::vector<size_t> nitems = {50, 50000, 5000000};
    std::vector<unsigned> plens(11);
    std::iota(plens.begin(), plens.end(), 6);
    for(const auto nitems: nitems) {
        for(const auto p: plens) {
            BBitMinHasher<uint64_t> bb(p, 40);
            BBitMinHasher<uint64_t> onebb(p, 1);
            BBitMinHasher<uint64_t> onebb2(p, 1);
            BBitMinHasher<uint64_t> onebb3(p, 1);
            BBitMinHasher<uint64_t> bb2(p, 40);
            BBitMinHasher<uint64_t> bb3(p, 40);
            BBitMinHasher<uint64_t> bb4(p, 40);
            for(size_t i = 0; i < nitems; ++i) {
                bb.addh(i);
                bb3.addh(i + nitems / 10);
                onebb.addh(i);
                onebb2.addh(i + nitems * 9 / 10);
                onebb3.addh(i + nitems / 10);
                bb2.addh(i + nitems / 2);
                bb4.addh(i + nitems);
            }
            bb.densify(); bb2.densify(); bb3.densify();
            auto f1 = bb.finalize(), f2 = bb2.finalize(), f3 = bb3.finalize(), f4 = bb4.finalize();
            auto neq12 = bb.nmatches(bb2), neq13 = bb.nmatches(bb3), neq14 = bb.nmatches(bb4);
            auto bbneq12 = f1.nmatches(f2), bbneq13 = f1.nmatches(f3), bbneq14 = f1.nmatches(f4);
#if VERBOSE_AF
            std::fprintf(stderr, "before b-bit: %zu [1,2], %zu [1,3]\n", neq12, neq13);
            std::fprintf(stderr, "after b-bit: %zu [1,2], %zu [1,3]\n", bbneq12, bbneq13);
#endif
            assert(neq12 == bbneq12);
            assert(neq13 == bbneq13);
            assert(neq14 == bbneq14);
            onebb.densify(); onebb2.densify();
            auto fl1 = onebb.finalize(), fl2 = onebb2.finalize(), fl3 = onebb3.finalize();
            if(superverbose) {
                std::fprintf(stderr, "[p=%u,b=1,n=%zu] ji: %f (should be %f). one-permutation estimate: %f\n", p, nitems, fl1.jaccard_index(fl2), 1./19, onebb.jaccard_index(onebb2));
                std::fprintf(stderr, "[p=%u,b=1,n=%zu] ji: %f (should be %f). one-permutation estimate: %f\n", p, nitems, fl1.jaccard_index(fl3), 9./11, onebb.jaccard_index(onebb3));
            }
        }
    }
}

void verify_popcount() {
    BBitMinHasher<uint64_t> b1(10, 4), b2(10, 4);
    b1.addh(1);
    b1.addh(4);
    b1.addh(137);

    b2.addh(1);
    b2.addh(4);
    b2.addh(17);
    auto f1 = b1.cfinalize(), f2 = b2.cfinalize();
VERBOSE_ONLY(
    std::fprintf(stderr, "f1 popcount: %" PRIu64 "\n", f1.popcnt());
    std::fprintf(stderr, "f2 popcount: %" PRIu64 "\n", f2.popcnt());
)
#if 0
    b1.show();
    b2.show();
#endif
    auto b3 = b1 + b2;
    //b3.show();
    auto f3 = b3.finalize();
    VERBOSE_ONLY(std::fprintf(stderr, "f3 popcount: %" PRIu64 "\n", f3.popcnt());)
    VERBOSE_ONLY(std::fprintf(stderr, "eqb: %zu. With itself: %zu\n", size_t(f1.equal_bblocks(f2)), size_t(f1.equal_bblocks(f1)));)
}

int main(int argc, char *argv[]) {
    superverbose = std::find_if(argv, argv + argc, [](auto x) {return std::strcmp(x, "--superverbose") == 0;}) != argv + argc;
    verify_correctness();
    verify_popcount();
    ICWSampler<float, uint64_t> sampler(1024);
    const unsigned long long niter = argc == 1 ? 5000000uLL: std::strtoull(argv[1], nullptr, 10);

    for(size_t i = 7; i < 15; i += 2) {
        for(const auto b: {13u, 32u, 8u, 7u, 14u, 17u, 3u, 1u}) {
            std::fprintf(stderr, "b: %u. i: %zu\n", b, i);
            SuperMinHash<policy::SizePow2Policy> smhp2(1 << i);
            SuperMinHash<policy::SizeDivPolicy>  smhdp(1 << i);
            SuperMinHash<policy::SizePow2Policy> smhp21(1 << i);
            SuperMinHash<policy::SizeDivPolicy>  smhdp1(1 << i);
            hll::hll_t h1(i), h2(i);
            uint64_t seed = h1.hash(h1.hash(i) ^ h1.hash(b));
            using HasherType = hash::WangHash;
            BBitMinHasher<uint64_t, HasherType> b1(i, b, 1, seed), b2(i, b, 1, seed), b3(i, b, 1, seed);
            std::unique_ptr<BBitMinHasher<uint64_t, HasherType>> b1_smaller(new BBitMinHasher<uint64_t, HasherType>(i - 4, b, 1, seed));
            size_t dbval = 1.5 * (size_t(1) << i);
            DivBBitMinHasher<uint64_t> db1(dbval, b), db2(dbval, b), db3(dbval, b);
            //DivBBitMinHasher<uint64_t> fb(i, b);
            CountingBBitMinHasher<uint64_t, uint32_t> cb1(i, b), cb2(i, b), cb3(i, b);
            DefaultRNGType gen(137 + (i * b));
            size_t shared = 0, b1c = 0, b2c = 0;
            for(size_t i = niter; --i;) {
                auto v = gen();
                switch(v & 0x3uL) {
                    case 0:
                    case 1: h1.addh(v); h2.addh(v);
                            b2.addh(v); b1.addh(v); ++shared;
                            b3.addh(v);
                            db1.addh(v); db2.addh(v);
                            smhp2.addh(v); smhp21.addh(v);
                            smhdp.addh(v); smhdp1.addh(v);
                            if(b1_smaller) b1_smaller->addh(v);
                    /*fb.addh(v);*/
                    break;
                    case 2: h1.addh(v); b1.addh(v); ++b1c; b3.addh(v); cb3.addh(v); db1.addh(v);
                            smhp2.addh(v);
                            smhdp.addh(v);
                            if(b1_smaller) b1_smaller->addh(v);
                    break;
                    case 3: h2.addh(v); b2.addh(v); ++b2c; cb1.addh(v); db2.addh(v);
                            smhdp1.addh(v);
                            smhp21.addh(v);
                    break;
                }
                //if(i % 250000 == 0) std::fprintf(stderr, "%zu iterations left\n", size_t(i));
            }
            {
                auto comp = b1.compress(i - 4);
                assert(b1_smaller);
                auto &precomp = *b1_smaller.get();
                assert(precomp.size() == comp.size());
                bool fail = !std::equal(comp.core().begin(), comp.core().end(), precomp.core().begin());
                if(fail) {
                    for(size_t i = 0, e = comp.size(); i < e; ++i)
                        std::fprintf(stderr, "index: %zu. lhs: %zu. rhs: %zu. diff: %d\n", i, size_t(comp.core()[i]), size_t(precomp.core()[i]), int(comp.core()[i] - precomp.core()[i]));
                    assert(!fail);
                }
                //assert(std::equal(comp.core().begin(), comp.core().end(), precomp.core().begin()));
            }
            b1.densify();
            b2.densify();
            auto est = (b1 + b2).cardinality_estimate();
            auto usest = b1.union_size(b2);
            VERBOSE_ONLY(std::fprintf(stderr, "union est by union: %f. by union_size: %f. difference: %12e\n", est, usest, (est - usest));)
            assert(std::abs(est - usest) < 1e-6);
            assert(est == usest);
            auto f1 = b1.finalize(), f2 = b2.finalize(), f3 = b3.finalize();
            assert(i <= 9 || std::abs(est - niter) < niter * .1 || !std::fprintf(stderr, "est: %lf. niter: %zu\n", est, size_t(niter)));
            //b1 += b2;
            auto f12 = b1.finalize();
            auto fdb1 = db1.finalize();
            auto fdb2 = db2.finalize();
            auto smh1 = smhp2.finalize(16), smh2 = smhp21.finalize(16);
            auto smhd1 = smhdp.finalize(16), smhd2 = smhdp1.finalize(16);
            auto smh1ji = smh1.jaccard_index(smh1);
            VERBOSE_ONLY(std::fprintf(stderr, "smh1ji: %g\n", smh1ji);)
            assert(smh1ji == 1.);
            auto pji = smh1.jaccard_index(smh2);
            VERBOSE_ONLY(std::fprintf(stderr, "estimate: %f. nmin: %u. b: %u\n", pji, 1u << i, b);)
            if(std::abs(pji - .5)  > 0.05) {
                std::fprintf(stderr, "original (no b-bit): %f\n", b1.jaccard_index(b2));
                std::fprintf(stderr, ">.05 error: estimate: %f. nmin: %u. b: %u. %f%% error\n", pji, 1u << i, b, std::abs(pji - .5) / .5 * 100);
            }
            assert(std::abs(smh1.jaccard_index(smh2) - .5) < 0.1 || i <= 7);

            if(superverbose) {
                std::fprintf(stderr, "with ss=%zu, smh1 and itself: %lf. 2 and 2/1 jaccard? %lf/%lf\n", size_t(1) << i, double(smh1.jaccard_index(smh1)), double(smh2.jaccard_index(smh1)), smh1.jaccard_index(smh2));
                std::fprintf(stderr, "smh1 card %lf, smh2 %lf\n", smh1.est_cardinality_, smh2.est_cardinality_);
                std::fprintf(stderr, "with ss=%zu, smhd1 and itself: %lf. 2 and 2/1 jaccard? %lf/%lf\n", size_t(1) << i, double(smhd1.jaccard_index(smhd1)), double(smhd2.jaccard_index(smhd1)), smhd1.jaccard_index(smhd2));
                std::fprintf(stderr, "Expected Cardinality [shared:%zu/b1:%zu/b2:%zu]\n", shared, b1c, b2c);
                std::fprintf(stderr, "h1 est %lf, h2 est: %lf\n", h1.report(), h2.report());
                std::fprintf(stderr, "Estimate Harmonicard [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(HARMONIC_MEAN), b2.cardinality_estimate(HARMONIC_MEAN));
                std::fprintf(stderr, "Estimate div Harmonicard [b1:%lf/b2:%lf]\n", db1.cardinality_estimate(), db2.cardinality_estimate());
                std::fprintf(stderr, "Estimate HLL [b1:%lf/b2:%lf/b3:%lf]\n", b1.cardinality_estimate(HLL_METHOD), b2.cardinality_estimate(HLL_METHOD), b3.cardinality_estimate(HLL_METHOD));
                std::fprintf(stderr, "Estimate arithmetic mean [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(ARITHMETIC_MEAN), b2.cardinality_estimate(ARITHMETIC_MEAN));
                std::fprintf(stderr, "Estimate (median) b1:%lf/b2:%lf]\n", b1.cardinality_estimate(MEDIAN), b2.cardinality_estimate(MEDIAN));
                std::fprintf(stderr, "Estimate geometic mean [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(GEOMETRIC_MEAN), b2.cardinality_estimate(GEOMETRIC_MEAN));
                std::fprintf(stderr, "JI for f3 and f2: %lf\n", f1.jaccard_index(f2));
                std::fprintf(stderr, "JI for fdb1 and fdb2: %lf, where nmin = %zu and b = %d\n", fdb2.jaccard_index(fdb1), i, b);
                std::fprintf(stderr, "f1, f2, and f3 cardinalities: %lf, %lf, %lf\n", f1.est_cardinality_, f2.est_cardinality_, f3.est_cardinality_);
            }
            auto fcb1 = cb1.finalize(), fcb2 = cb3.finalize();
            //auto cb13res = fcb1.histogram_sums(fcb2);
            //assert(sizeof(cb13res) == sizeof(uint64_t) * 4);
            //std::fprintf(stderr, "cb13res %lf, %lf\n", cb13res.weighted_jaccard_index(), cb13res.jaccard_index());
            cb1.finalize().write("ZOMG.cb");
            decltype(cb1.finalize()) cbr("ZOMG.cb");
            auto deleter = []() {if(std::system("rm ZOMG.cb")) throw std::runtime_error("Failed to delete ZOMG.cb");};
            scope_executor<decltype(deleter)> se(deleter);
            assert(cbr == cb1.finalize());
            //cbr.histogram_sums(cb2.finalize()).print();
            auto whl = b1.make_whll();
            auto phl = b1.make_packed16hll();
            std::fprintf(stderr, "p16 card: %lf\n", phl.cardinality_estimate());
            std::fprintf(stderr, "whl card: %lf/%zu vs expected %lf/%lf/%lf\n", whl.cardinality_estimate(), whl.core_.size(), f1.est_cardinality_, h1.report(), whl.union_size(whl));
        }
    }
    BBitMinHasher<uint64_t> notfull(12);
    for(size_t i = 0; i < 4000; ++i) {
        notfull.addh(i);
    }
    auto wideh = notfull.make_whll();
    std::fprintf(stderr, "wideh estimate: %g.\n", wideh.cardinality_estimate());
    auto regh = notfull.make_hll();
    regh.sum();
    std::fprintf(stderr, "hll estimate: %g.\n", regh.report());
    regh.set_estim(hll::ORIGINAL);
    regh.not_ready();
    regh.sum();
    std::fprintf(stderr, "hll estimate: %g.\n", regh.report());
}
