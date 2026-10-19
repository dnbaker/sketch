#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>
#include "aesctr/wy.h"
#include "sketch/isz.h"
#include "sketch/mh.h"

using sketch::isz::intersection_size_sets;
using sketch::isz::intersection_size_sets_min;
using sketch::isz::detail::isz_merge_count;

static std::uint64_t checked = 0, gate_failures = 0;

template<typename T>
static void check_capped(const T *a, std::size_t na, const T *b, std::size_t nb,
                         std::uint64_t min_count, std::uint64_t exact, const char *ctx) {
    const std::uint64_t got = intersection_size_sets_min(a, na, b, nb, min_count);
    ++checked;
    // Threshold contract: v <= exact and (v >= min_count) == (exact >= min_count).
    const bool ok = (got <= exact) && ((got >= min_count) == (exact >= min_count));
    if(!ok) {
        ++gate_failures;
        std::fprintf(stderr, "[GATE FAIL] capped %s: min=%llu exact=%llu got=%llu (na=%zu nb=%zu)\n",
                     ctx, (unsigned long long)min_count, (unsigned long long)exact,
                     (unsigned long long)got, na, nb);
        if(gate_failures > 5) { std::fprintf(stderr, "too many failures, aborting\n"); std::exit(1); }
    }
}

template<typename T>
static void check_thresholds(const T *a, std::size_t na, const T *b, std::size_t nb,
                             std::uint64_t exact, const char *ctx) {
    for(const std::uint64_t t: {std::uint64_t(0), std::uint64_t(1), std::uint64_t(exact / 2),
                                std::uint64_t(exact), std::uint64_t(exact + 1), std::uint64_t(na + nb + 1)}) {
        check_capped(a, na, b, nb, t, exact, ctx);
    }
}

template<typename T>
static void check_one(const T *a, std::size_t na, const T *b, std::size_t nb, const char *ctx) {
    const std::uint64_t ref = isz_merge_count(a, na, b, nb);
    const std::uint64_t got = intersection_size_sets(a, na, b, nb);
    ++checked;
    if(ref != got) {
        ++gate_failures;
        std::fprintf(stderr, "[GATE FAIL] %s: ref=%llu got=%llu (na=%zu nb=%zu)\n",
                     ctx, (unsigned long long)ref, (unsigned long long)got, na, nb);
        std::fprintf(stderr, "  a: ");
        for(std::size_t i = 0; i < na; ++i) std::fprintf(stderr, "%llu ", (unsigned long long)a[i]);
        std::fprintf(stderr, "\n  b: ");
        for(std::size_t i = 0; i < nb; ++i) std::fprintf(stderr, "%llu ", (unsigned long long)b[i]);
        std::fprintf(stderr, "\n");
        if(gate_failures > 5) { std::fprintf(stderr, "too many failures, aborting\n"); std::exit(1); }
    }
    check_thresholds(a, na, b, nb, ref, ctx);
}

template<typename T>
static void check_vec(const std::vector<T> &a, const std::vector<T> &b, const char *ctx) {
    check_one(a.data(), a.size(), b.data(), b.size(), ctx);
}

template<typename T, typename RNG>
static std::vector<T> random_set(std::size_t n, std::uint64_t universe, RNG &rng) {
    std::vector<T> v;
    v.reserve(n);
    ska::flat_hash_set<T> seen;
    while(v.size() < n) {
        const T x = T(rng() % universe);
        if(seen.insert(x).second) v.push_back(x);
    }
    std::sort(v.begin(), v.end());
    return v;
}

template<typename T>
static void gate_exhaustive_small() {
    // All pairs of subsets of {0..7}: 256 subsets, 65536 pairs.
    for(unsigned ma = 0; ma < 256; ++ma) {
        std::vector<T> a;
        for(unsigned i = 0; i < 8; ++i) if(ma & (1u << i)) a.push_back(T(i * 2));
        for(unsigned mb = 0; mb < 256; ++mb) {
            std::vector<T> b;
            for(unsigned i = 0; i < 8; ++i) if(mb & (1u << i)) b.push_back(T(i * 2));
            check_vec(a, b, "exhaustive-subsets");
        }
    }
}

template<typename T>
static void gate_adversarial() {
    const std::vector<std::pair<std::vector<T>, std::vector<T>>> fixed = {
        {{}, {}}, {{}, {T(3)}}, {{T(3)}, {}}, {{T(1)}, {T(1)}}, {{T(1)}, {T(2)}},
        {{T(2)}, {T(1)}}, {{T(1), T(3)}, {T(2)}}, {{T(1)}, {T(1), T(2)}},
        {{T(5)}, {T(5)}}, {{T(0)}, {T(0), T(1)}},
    };
    for(const auto &pr: fixed) check_vec(pr.first, pr.second, "fixed");
    // Equal disjoint / subset / superset block patterns around SIMD block
    // boundaries (4 and 8) and one past.
    for(std::size_t n: {3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 4095, 4096, 4097}) {
        std::vector<T> a, b, c, d;
        for(std::size_t i = 0; i < n; ++i) {
            a.push_back(T(2 * i));        // evens
            b.push_back(T(2 * i + 1));    // odds (disjoint)
            c.push_back(T(i));            // interleaved 1-apart with a
        }
        for(std::size_t i = 0; i < n / 2; ++i) d.push_back(T(4 * i)); // subset-ish of a
        check_vec(a, b, "disjoint");
        check_vec(a, c, "1-apart");
        check_vec(a, d, "subset");
        check_vec(d, a, "superset");
        check_vec(a, a, "self");
    }
    // Skew sweep: short list vs long list.
    for(std::size_t nb: {1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}) {
        std::vector<T> a = {T(2), T(9), T(17), T(33), T(100), T(200), T(511), T(4097)};
        std::vector<T> b;
        for(std::size_t i = 0; i < nb; ++i) b.push_back(T(3 * i + 1));
        check_vec(a, b, "skew-a-short");
        check_vec(b, a, "skew-b-short");
    }
}

template<typename T>
static void gate_multiset() {
    // Multiset inputs (e.g. FinalCRMinHash registers padded with repeated
    // T::max): multiplicity semantics must be exact, exercising the
    // duplicate-scan fallback. Reference: the pairwise-min-multiplicity
    // merge, computed here by std::set intersection on the unique values
    // with multiplicities taken into account by direct counting.
    auto ref_multiplicity = [](const std::vector<T> &a, const std::vector<T> &b) {
        std::uint64_t ret = 0;
        std::size_t i = 0, j = 0;
        while(i < a.size() && j < b.size()) {
            if(a[i] == b[j]) { ++ret; ++i; ++j; }
            else if(a[i] < b[j]) ++i;
            else ++j;
        }
        return ret;
    };
    const T M = std::numeric_limits<T>::max();
    const std::vector<std::pair<std::vector<T>, std::vector<T>>> cases = {
        {{M, M, M}, {M, M}},                    // padded-counting-sketch shape
        {{M, M, M}, {M}},
        {{1, 2, 2, 2, 7}, {2, 2, 9}},           // heavy duplicates
        {{2, 2, 2, 2}, {1, 2, 2, 3}},           // FESISA-would-miscount case
        {{1, 1, 3, 3}, {1, 1, 1, 3}},
        {{5, 5}, {5, 5, 5, 5, 5}},
        {{M, M}, {M, M, M, M}},
    };
    for(const auto &pr: cases) {
        const std::uint64_t ref = ref_multiplicity(pr.first, pr.second);
        const std::uint64_t got = intersection_size_sets(pr.first.data(), pr.first.size(), pr.second.data(), pr.second.size());
        ++checked;
        if(ref != got) {
            std::fprintf(stderr, "[GATE FAIL] multiset: ref=%llu got=%llu\n",
                         (unsigned long long)ref, (unsigned long long)got);
            if(++gate_failures > 5) std::exit(1);
        }
        // Capped path on multisets: threshold contract must still hold.
        for(const std::uint64_t t: {std::uint64_t(0), std::uint64_t(1), ref / 2, ref, ref + 1}) {
            const std::uint64_t gv = intersection_size_sets_min(pr.first.data(), pr.first.size(),
                                                                pr.second.data(), pr.second.size(), t);
            ++checked;
            const bool ok = (gv <= ref) && ((gv >= t) == (ref >= t));
            if(!ok) {
                std::fprintf(stderr, "[GATE FAIL] multiset capped: t=%llu ref=%llu got=%llu\n",
                             (unsigned long long)t, (unsigned long long)ref, (unsigned long long)gv);
                if(++gate_failures > 5) std::exit(1);
            }
        }
    }
    // Random multiset sweeps: sorted vectors with planted duplicates.
    wy::WyHash<> rng(4242);
    for(unsigned trial = 0; trial < 5000; ++trial) {
        const std::size_t na = 1 + rng() % 512, nb = 1 + rng() % 512;
        std::vector<T> a, b;
        for(std::size_t i = 0; i < na; ++i) a.push_back(T(rng() % 32));  // tiny universe -> many dups
        for(std::size_t i = 0; i < nb; ++i) b.push_back(T(rng() % 32));
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        const std::uint64_t ref = ref_multiplicity(a, b);
        const std::uint64_t got = intersection_size_sets(a.data(), a.size(), b.data(), b.size());
        ++checked;
        if(ref != got) {
            std::fprintf(stderr, "[GATE FAIL] random multiset: ref=%llu got=%llu\n",
                         (unsigned long long)ref, (unsigned long long)got);
            if(++gate_failures > 5) std::exit(1);
        }
        const std::uint64_t t = ref / 2;
        const std::uint64_t gv = intersection_size_sets_min(a.data(), a.size(), b.data(), b.size(), t);
        ++checked;
        if(gv > ref || (gv >= t) != (ref >= t)) {
            std::fprintf(stderr, "[GATE FAIL] random multiset capped: t=%llu ref=%llu got=%llu\n",
                         (unsigned long long)t, (unsigned long long)ref, (unsigned long long)gv);
            if(++gate_failures > 5) std::exit(1);
        }
    }
}

template<typename T, typename RNG>
static void gate_random(RNG &rng) {
    for(unsigned trial = 0; trial < 20000; ++trial) {
        const std::size_t na = rng() % 8192, nb = rng() % 8192;
        auto a = random_set<T>(na, (1u << 24) + 61, rng);
        auto b = random_set<T>(nb, (1u << 24) + 61, rng);
        check_vec(a, b, "random");
    }
}

static void integration_minhash() {
    // FinalRMinHash-level identity: SIMD dispatch must match brute force.
    wy::WyHash<uint64_t> rng(1337);
    for(const std::size_t k: {5, 17, 256, 1024, 8192}) {
        for(unsigned rep = 0; rep < 200; ++rep) {
            auto va = random_set<std::uint64_t>(k, (1ull << 40), rng);
            auto vb = random_set<std::uint64_t>(k, (1ull << 40), rng);
            sketch::FinalRMinHash<std::uint64_t> ra(va.begin(), va.end()), rb(vb.begin(), vb.end());
            const std::size_t is = ra.intersection_size(rb);
            const std::uint64_t ref = isz_merge_count(va.data(), va.size(), vb.data(), vb.size());
            if(is != ref) {
                std::fprintf(stderr, "[GATE FAIL] FinalRMinHash::intersection_size k=%zu: ref=%llu got=%zu\n",
                             k, (unsigned long long)ref, is);
                std::exit(1);
            }
            // jaccard_index is the exact full-merge register overlap.
            const double j = ra.jaccard_index(rb);
            const double jref = double(ref) / k;
            if(j != jref) {
                std::fprintf(stderr, "[GATE FAIL] FinalRMinHash::jaccard_index k=%zu: ref=%f got=%f\n", k, jref, j);
                std::exit(1);
            }
            // Thresholded interfaces: capped count contract + exact jaccard threshold test.
            for(const std::uint64_t t: {std::uint64_t(0), std::uint64_t(1), std::uint64_t(ref / 2), std::uint64_t(ref), std::uint64_t(ref + 1), std::uint64_t(k + 1)}) {
                const std::uint64_t got = sketch::isz::intersection_size_sets_min(
                    va.data(), va.size(), vb.data(), vb.size(), t);
                const bool ok = (got <= ref) && ((got >= t) == (ref >= t));
                if(!ok) {
                    std::fprintf(stderr, "[GATE FAIL] intersection_size_sets_min k=%zu t=%llu ref=%llu got=%llu\n",
                                 k, (unsigned long long)t, (unsigned long long)ref, (unsigned long long)got);
                    std::exit(1);
                }
                ++checked;
            }
            for(const double t: {0.05, 0.2, 0.5, 0.8, 0.99}) {
                const bool jl = ra.jaccard_at_least(rb, t);
                if(jl != (jref >= t)) {
                    std::fprintf(stderr, "[GATE FAIL] jaccard_at_least k=%zu t=%f jref=%f got=%d\n", k, t, jref, int(jl));
                    std::exit(1);
                }
                ++checked;
            }
            // Generic-container bounded path (std::set), same contract.
            const std::set<std::uint64_t> sa(va.begin(), va.end()), sb(vb.begin(), vb.end());
            for(const std::uint64_t t: {std::uint64_t(0), std::uint64_t(ref / 2), std::uint64_t(ref), std::uint64_t(ref + 1)}) {
                const std::uint64_t got = sketch::isz::intersection_size_min(sa, sb, t);
                const bool ok = (got <= ref) && ((got >= t) == (ref >= t));
                if(!ok) {
                    std::fprintf(stderr, "[GATE FAIL] isz::intersection_size_min(set) k=%zu t=%llu ref=%llu got=%llu\n",
                                 k, (unsigned long long)t, (unsigned long long)ref, (unsigned long long)got);
                    std::exit(1);
                }
                ++checked;
            }
            ++checked;
        }
    }
    // RangeMinHash-level smoke: streaming std::set sketch, thresholded
    // delegates must agree with the plain jaccard walk.
    {
        sketch::RangeMinHash<std::uint64_t> rm1(512), rm2(512);
        for(unsigned i = 0; i < 20000; ++i) {
            const std::uint64_t x = rng();
            rm1.addh(x);
            if(i % 5) rm2.addh(x); // ~80% shared stream
        }
        const double j = rm1.jaccard_index(rm2);
        for(const double t: {0.05, 0.2, 0.5, 0.8, 0.99}) {
            if(rm1.jaccard_at_least(rm2, t) != (j >= t)) {
                std::fprintf(stderr, "[GATE FAIL] RangeMinHash::jaccard_at_least t=%f j=%f\n", t, j);
                std::exit(1);
            }
            ++checked;
        }
        const std::uint64_t exact = rm1.intersection_size(rm2);
        for(const std::uint64_t t: {std::uint64_t(0), exact / 2, exact, exact + 1}) {
            const std::uint64_t got = rm1.intersection_size_min(rm2, t);
            const bool ok = (got <= exact) && ((got >= t) == (exact >= t));
            if(!ok) {
                std::fprintf(stderr, "[GATE FAIL] RangeMinHash::intersection_size_min t=%llu exact=%llu got=%llu\n",
                             (unsigned long long)t, (unsigned long long)exact, (unsigned long long)got);
                std::exit(1);
            }
            ++checked;
        }
    }
}

template<typename F>
static double time_median(F &&f, unsigned reps = 9) {
    double best = 1e300;
    for(unsigned r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        f();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        if(ns < best) best = ns;
    }
    return best;
}

static volatile std::uint64_t bench_sink;

template<typename F, typename P>
static double time_batch(F &&fn, P p1, P p2) {
    // macOS chrono granularity is ~1us: time batches and divide. The asm
    // barrier marks the input pointers (and accumulator) as clobbered each
    // iteration so the compiler cannot hoist the pure call out of the batch
    // or strength-reduce the accumulation.
    constexpr unsigned reps = 20000;
    return time_median([&]{
        std::uint64_t acc = 0;
        P a = p1, b = p2;
        for(unsigned r = 0; r < reps; ++r) {
            acc += fn(a, b);
            asm volatile("" : "+r"(a), "+r"(b), "+r"(acc));
        }
        bench_sink = acc;
    }, 5) / reps;
}

template<typename T>
static void bench_case(const char *label, const std::vector<T> &base, const std::vector<T> &b,
                       std::size_t k, std::uint64_t minc) {
    const T *based = base.data();
    const double t_scalar = time_batch([&](const T *a, const T *bb){ return isz_merge_count(a, k, bb, k); }, based, b.data());
    const double t_simd = time_batch([&](const T *a, const T *bb){ return intersection_size_sets(a, k, bb, k); }, based, b.data());
    const double t_simd_min = time_batch([&](const T *a, const T *bb){ return intersection_size_sets_min(a, k, bb, k, minc); }, based, b.data());
    std::fprintf(stderr, "  %-30s k=%5zu  scalar %9.1f ns   simd %9.1f ns   speedup %5.2fx   | capped %9.1f ns (%.2fx vs full simd)\n",
                 label, k, t_scalar, t_simd, t_scalar / t_simd, t_simd_min, t_simd / t_simd_min);
}

template<typename T, typename RNG>
static void bench(RNG &rng, const char *tag) {
    std::fprintf(stderr, "u%s (%s), best-of-5 batches of 20000, ns per intersection\n", sizeof(T) == 8 ? "64" : "32", tag);
    const T shift = sizeof(T) == 8 ? T(1ull << 33) : T(1u << 31);
    for(const std::size_t k: {128, 256, 1024, 8192}) {
        auto base = random_set<T>(k, std::uint64_t(shift), rng);
        // Realistic LSH-filtering fixture: ~50% overlap INTERLEAVED
        // irregularly (half of b drawn from a's values, half fresh) -- the
        // branch-hostile case for a scalar merge.
        std::vector<T> b = base;
        for(std::size_t i = 0; i < k / 2; ++i) {
            b[rng() % k] = T(rng() % shift); // overwrite half at random positions
        }
        std::sort(b.begin(), b.end());
        b.erase(std::unique(b.begin(), b.end()), b.end());
        while(b.size() < k) { // top up if the overwrite collided
            const T x = T(rng() % shift);
            if(!std::binary_search(b.begin(), b.end(), x)) b.insert(std::lower_bound(b.begin(), b.end(), x), x);
        }
        bench_case("interleaved ~50% (capped@75%)", base, b, k, k - k / 4);
        bench_case("interleaved ~50% (capped@90%)", base, b, k, k - k / 10);
        // Prefix-shared fixture: b shares its head with a then diverges --
        // the branch-friendly worst case for SIMD (published FESISA loss regime).
        std::vector<T> c = base;
        for(std::size_t i = k / 2; i < k; ++i) c[i] = T(base[i] + shift + i);
        std::sort(c.begin(), c.end());
        bench_case("prefix-shared (capped@75%)", base, c, k, k - k / 4);
    }
}

int main(int argc, char **argv) {
    wy::WyHash<uint64_t> rng(20261019);
    gate_exhaustive_small<std::uint32_t>();
    gate_exhaustive_small<std::uint64_t>();
    gate_adversarial<std::uint32_t>();
    gate_adversarial<std::uint64_t>();
    gate_multiset<std::uint32_t>();
    gate_multiset<std::uint64_t>();
    gate_random<std::uint32_t>(rng);
    gate_random<std::uint64_t>(rng);
    integration_minhash();
    std::fprintf(stderr, "gate: %llu checks, %llu failures\n", (unsigned long long)checked, (unsigned long long)gate_failures);
    if(gate_failures) {
        std::fprintf(stderr, "GATE FAILED -- refusing to print timings\n");
        return 1;
    }
    if(argc > 1 && std::string(argv[1]) == "--bench") {
        bench<std::uint32_t>(rng, "u32");
        bench<std::uint64_t>(rng, "u64");
    } else {
        std::fprintf(stderr, "(pass --bench for the timing table)\n");
    }
    std::fprintf(stderr, "passed!\n");
    return 0;
}
