// Implementations of tests/reference.h — ALL CPU-sketch-header code for the
// GPU bit-identity tests lives in this plain-g++ TU (nvcc must never see
// include/sketch/*.h). Compiled by $(CXX) per gpu/Makefile; linked into the
// test binary by nvcc alongside sketch.o.
#include "reference.h"

#include <cstring>
#include <random>
#include <vector>

#include "hash.h"
#include "hll.h"
#include "bf.h"
#include "ccm.h"
#include "update.h"
#include "bbmh.h"
#include "fracminhash.h"

namespace skref {

std::uint64_t splitmix64(std::uint64_t x) {
    return sketch::hash::SplitMix64Hash()(x);
}

std::uint32_t bf_ngroups(std::uint32_t l2sz, std::uint32_t nh) {
    sketch::bf_t ref(l2sz, nh, 137);
    return std::uint32_t(ref.seeds().size());
}

void bf(const std::uint64_t *els, std::size_t n, std::uint32_t l2sz, std::uint32_t nh,
        std::uint64_t *flat, std::uint64_t *core) {
    sketch::bf_t ref(l2sz, nh, 137);
    const auto &seeds = ref.seeds();
    const std::size_t ng = seeds.size();
    sketch::hash::WangHash wh;
    for(std::size_t e = 0; e < n; ++e) {
        ref.addh(els[e]);
        for(std::size_t g = 0; g < ng; ++g)
            flat[e * ng + g] = wh(els[e] ^ seeds[g]);
    }
    std::memcpy(core, ref.core().data(), ref.core().size() * sizeof(std::uint64_t));
}

void hll_update(const std::uint64_t *hashes, std::size_t n, std::uint32_t np, std::uint8_t *core) {
    sketch::hll::hll_t ref(np);
    for(std::size_t i = 0; i < n; ++i) ref.add(hashes[i]);
    std::memcpy(core, ref.core().data(), ref.core().size());
}

void hll_update_hashed(const std::uint64_t *in, std::size_t n, std::uint32_t np, std::uint8_t *core) {
    sketch::hll::hllbase_t<sketch::hash::SplitMix64Hash> ref(np);
    for(std::size_t i = 0; i < n; ++i) ref.addh(in[i]);
    std::memcpy(core, ref.core().data(), ref.core().size());
}

void hll_merge3(const std::uint64_t *s0, const std::uint64_t *s1, const std::uint64_t *s2,
                std::size_t n, std::uint32_t np, std::uint8_t *core, double *report) {
    sketch::hll::hll_t h0(np), h1(np), h2(np);
    for(std::size_t i = 0; i < n; ++i) {
        h0.add(s0[i]);
        h1.add(s1[i]);
        h2.add(s2[i]);
    }
    h0 += h1;
    h0 += h2;
    std::memcpy(core, h0.core().data(), h0.core().size());
    *report = h0.report();
}

double hll_report(const std::uint8_t *core, std::uint32_t np) {
    sketch::hll::hll_t ref(np);
    std::memcpy(ref.mutable_core().data(), core, std::size_t(1) << np);
    return ref.report();
}

void ccm(const std::uint64_t *els, std::size_t n, std::uint32_t l2sz, std::uint32_t nh,
         std::uint64_t seed, std::uint32_t nbits, std::uint64_t *flat, std::uint32_t *counters) {
    using Ref = sketch::ccmbase_t<sketch::update::Increment, sketch::DefaultCompactVectorType,
                                  sketch::hash::WangHash, false>;
    Ref ref(int(nbits), int(l2sz), nh, seed);
    std::mt19937_64 mt(seed + 4); // mirrors ccmbase_t's seed generation
    std::vector<std::uint64_t> seeds(nh);
    for(auto &s: seeds) s = mt();
    sketch::hash::WangHash wh;
    for(std::size_t e = 0; e < n; ++e) {
        ref.addh(els[e]);
        for(std::size_t i = 0; i < nh; ++i)
            flat[e * nh + i] = wh(els[e] ^ seeds[i]);
    }
    std::size_t j = 0;
    ref.for_each_register([&](const auto v) { counters[j++] = std::uint32_t(v); });
}

void bbmh(const std::uint32_t *sids, const std::uint64_t *hvs, std::size_t n,
          std::uint32_t p, std::uint32_t nsk, std::uint64_t *core) {
    std::vector<sketch::BBitMinHasher<std::uint64_t>> refs;
    refs.reserve(nsk);
    for(std::uint32_t s = 0; s < nsk; ++s) refs.emplace_back(p);
    for(std::size_t i = 0; i < n; ++i) refs[sids[i]].add(hvs[i]);
    for(std::uint32_t s = 0; s < nsk; ++s)
        std::memcpy(core + (std::size_t(s) << p), refs[s].core().data(), (std::size_t(1) << p) * sizeof(std::uint64_t));
}

std::size_t fmh(const std::uint32_t *sids, const std::uint64_t *hvs, std::size_t n,
                std::uint64_t max_hash, std::uint32_t nsk,
                std::uint64_t *kept_out, std::uint32_t *kept_sid_out,
                std::uint32_t *counts_out, std::uint64_t *mins_out) {
    std::vector<sketch::fracminhashbase_t<>> refs;
    refs.reserve(nsk);
    for(std::uint32_t s = 0; s < nsk; ++s) refs.emplace_back(max_hash);
    // add(), not addh(): the GPU consumes pre-hashed values, so the CPU
    // reference must see the same hash values (addh would hash again).
    for(std::size_t i = 0; i < n; ++i) refs[sids[i]].add(hvs[i]);
    std::size_t total = 0;
    for(std::uint32_t s = 0; s < nsk; ++s) {
        refs[s].for_each_kept([&](std::uint64_t v) {
            kept_out[total] = v;
            kept_sid_out[total] = s;
            ++total;
        });
        counts_out[s] = std::uint32_t(refs[s].size());
        mins_out[s] = refs[s].min_hash();
    }
    return total;
}

} // namespace skref
