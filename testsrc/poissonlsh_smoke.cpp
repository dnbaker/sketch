// poissonlsh_smoke: first testsrc coverage for include/sketch/poissonlsh.h
// (p-stable LSH over blaze; nothing else in the repo includes it). Scope:
//   1. determinism: two same-seed L2LSHasher instances hash identical points
//      to identical floor((a.x+b)/w) values; a different seed differs
//   2. LSHTable: self-retrieval appears in every one of the l bands, and
//      every returned candidate id is in range
//   3. LSH sanity (seeded, loose): close pairs (distance ~0.05) collide in
//      more bands than cross-cluster pairs (distance ~16), via the module's
//      own query() band-match counts
// The cloud is 2-D normal so the run stays trivial; cost is the blaze TU.
#include "poissonlsh.h"
#include <blaze/Math.h>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace sketch;
using namespace sketch::hash;

static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

int main() {
    const unsigned dim = 2, k = 4, l = 8, n = 128;
    const double w = 1.5;
    std::mt19937_64 rng(20261017);
    std::normal_distribution<double> gauss;
    // two clusters centered at (+/-8, 0); within each, close pairs
    std::vector<blaze::DynamicVector<double>> pts(n, blaze::DynamicVector<double>(dim));
    for(unsigned i = 0; i < n; i += 2) {
        const double cx = (i < n / 2) ? -8.: 8.;
        pts[i][0] = cx + gauss(rng);
        pts[i][1] = gauss(rng);
        pts[i + 1][0] = pts[i][0] + 0.05 * gauss(rng);
        pts[i + 1][1] = pts[i][1] + 0.05 * gauss(rng);
    }
    // ---- 1. same-seed determinism, different-seed divergence ----
    using H = L2LSHasher<double>;
    H h1(dim, k, l, w, 12345), h2(dim, k, l, w, 12345), h3(dim, k, l, w, 999);
    for(unsigned i = 0; i < 8; ++i) {
        const auto a = blaze::evaluate(h1.hash(pts[i]));
        const auto b = blaze::evaluate(h2.hash(pts[i]));
        const auto c = blaze::evaluate(h3.hash(pts[i]));
        CHECK(a.size() == k * l, "hash width %zu != k*l %u", a.size(), k * l);
        unsigned same = 0, diff = 0;
        for(size_t j = 0; j < a.size(); ++j) {
            CHECK(a[j] == b[j], "same-seed instances diverged at point %u coord %zu", i, j);
            same += a[j] == c[j]; ++diff;
        }
        CHECK(same != diff, "different seed produced an identical hash (point %u)", i);
    }
    std::fprintf(stderr, "poissonlsh same-seed determinism ok\n");
    // ---- 2. LSHTable: self-retrieval in every band, ids in range ----
    using T = LSHTable<H>; // IT=uint32_t ids, KT=uint64_t band keys
    T t(dim, k, l, w, 777);
    for(unsigned i = 0; i < n; ++i) t.add(pts[i], i);
    for(unsigned i = 0; i < n; ++i) {
        const auto m = t.query(pts[i]);
        CHECK(m.count(i) == 1 && m.at(i) == l, "self band-match count %u != l=%u (point %u)",
              m.count(i) ? m.at(i): 0, l, i);
        for(const auto &kv: m) CHECK(kv.first < n, "candidate id %u out of range (n=%u)", kv.first, n);
    }
    // determinism at the table level: same seed, same band-key answers
    T t2(dim, k, l, w, 777);
    for(unsigned i = 0; i < n; ++i) t2.add(pts[i], i);
    for(unsigned i = 0; i < 4; ++i) {
        const auto m1 = t.query(pts[i]), m2 = t2.query(pts[i]);
        CHECK(m1.size() == m2.size(), "table determinism: map sizes differ (point %u)", i);
        for(const auto &kv: m1)
            CHECK(m2.count(kv.first) == 1 && m2.at(kv.first) == kv.second,
                  "table determinism: band matches differ for id %u (point %u)", kv.first, i);
    }
    std::fprintf(stderr, "poissonlsh LSHTable self-retrieval/range ok\n");
    // ---- 3. neighbor collides more than distant (loose, seeded) ----
    uint64_t close_hits = 0, far_hits = 0;
    for(unsigned i = 0; i < n; i += 2) {
        const auto mc = t.query(pts[i]);
        if(mc.count(i + 1)) close_hits += mc.at(i + 1);
        const unsigned far = (i < n / 2) ? i + n / 2: i - n / 2; // other cluster
        const auto mf = t.query(pts[far]);
        if(mf.count(i + 1)) far_hits += mf.at(i + 1);
    }
    CHECK(close_hits > 0, "close pairs never collided (w=%g too small for dim %u?)", w, dim);
    CHECK(close_hits > far_hits, "LSH sanity failed: close-band matches %llu <= far %llu",
          (unsigned long long)close_hits, (unsigned long long)far_hits);
    std::fprintf(stderr, "poissonlsh LSH sanity ok (close %llu vs far %llu band matches)\n",
                 (unsigned long long)close_hits, (unsigned long long)far_hits);
    if(failures) {
        std::fprintf(stderr, "poissonlsh_smoke: %d FAILURES\n", failures);
        return 1;
    }
    std::fprintf(stderr, "all poissonlsh smoke tests passed\n");
    return 0;
}
