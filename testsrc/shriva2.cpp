#include "blaze/Math.h"
#include "sketch/mh.h"
#include <iostream>
#include <utility>
#include <map>

template<typename IT>
static INLINE auto count_paired_1bits(IT x) {
    static constexpr IT bitmask = static_cast<IT>(0x5555555555555555uLL);
    return sketch::popcount((x >> 1) & x & bitmask);
}

size_t shared_rems(const uint16_t *lhs, const uint16_t *rhs, size_t nh) {
    size_t ret = 0;
    const __m256i *lhp((const __m256i *)lhs);
    const __m256i *rhp((const __m256i *)rhs);
    while(nh >= sizeof(__m256i) / sizeof(uint16_t)) {
        ret += count_paired_1bits(_mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_loadu_si256(lhp++), _mm256_loadu_si256(rhp++))));
        nh -= sizeof(__m256i) / sizeof(uint16_t);
    }
    lhs = (const uint16_t *)lhp;
    rhs = (const uint16_t *)rhp;
    while(nh--)
        ret += *lhs++ == *rhs++;
    return ret;
}

size_t shared_rems(const uint32_t *lhs, const uint32_t *rhs, size_t nh) {
    size_t ret = 0;
    const __m256i *lhp((const __m256i *)lhs);
    const __m256i *rhp((const __m256i *)rhs);
    while(nh >= sizeof(__m256i) / sizeof(uint32_t)) {
        ret += sketch::popcount(_mm256_movemask_ps((__m256)_mm256_cmpeq_epi32(_mm256_loadu_si256(lhp++), _mm256_loadu_si256(rhp++))));
        nh -= sizeof(__m256i) / sizeof(uint32_t);
    }
    lhs = (const uint32_t *)lhp;
    rhs = (const uint32_t *)rhp;
    while(nh--)
        ret += *lhs++ == *rhs++;
    return ret;
}

using Sig = std::uint32_t;

#if SPARSE
#define ShrivastavaHash SparseShrivastavaHash
using RType = blaze::CompressedVector<float>;
#else
using RType = std::vector<float>;
#endif

//#define true false

int main(int argc, char **argv) {
    constexpr unsigned d = 33570;
    int nh = argc > 1 ? std::atoi(argv[1]): 50;
    const size_t nsamples = 5390;
    std::vector<RType> data(nsamples);
    std::FILE *fp = std::fopen("5kcells.dense.5390.33570.mat", "rb");
    if(!fp) throw 1;
    auto hashstart = std::chrono::high_resolution_clock::now();
    sketch::ShrivastavaHash<true, Sig> whasher(d, nh);
    auto hashstop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "hasher construction took %gms\n", std::chrono::duration<double, std::milli>(hashstop - hashstart).count());
#if SPARSE
    float buf[d];
#endif
    for(size_t i = 0; i < nsamples; ++i) {
        
        auto &nextrow(data[i]);
#if SPARSE
        nextrow = blaze::CompressedVector<float>(d);
        if(std::fread(buf, sizeof(float), d, fp) != d) throw std::runtime_error("Failed to read row");
        for(size_t i = 0; i < d; ++i) {
            if(buf[i] > 0)
                 nextrow.set(i, buf[i]);
        }
#else
        nextrow.resize(d);
        if(std::fread(nextrow.data(), sizeof(float), d, fp) != d)
            throw std::runtime_error("Failed to read row");
#endif
        std::fprintf(stderr, "loaded %zu/%zu\n", i + 1, nsamples);
    }
    for(auto &r: data) {
#if SPARSE
        r = sqrt(r);
#else
        std::transform(r.begin(), r.end(), r.begin(), [](auto x){if(!x) return 0.f; return std::sqrt(x);});
#endif
    }
    whasher.set_threshold(7); // Maximum value
    using Result = std::decay_t<decltype(whasher.hash(data[0]))>;
    std::vector<Result> hashes(data.size());
    std::vector<float> mstimes(data.size());
    auto start = std::chrono::high_resolution_clock::now();
    size_t completed = 0;
    OMP_PFOR
    for(size_t i = 0; i < data.size(); ++i) {
        const auto &dat(data[i]);
        auto start = std::chrono::high_resolution_clock::now();
        hashes[i] = std::move(whasher.hash(dat));
        auto stop = std::chrono::high_resolution_clock::now();
        auto t = std::chrono::duration<double, std::milli>(stop - start).count();
        mstimes[completed++] = t;
        if(completed % 256 == 0) std::fprintf(stderr, "completed %zu/%zu\n", completed, nsamples);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Hashing took %g ms\n", std::chrono::duration<double, std::milli>(stop - start).count());
    std::vector<float> wsimilarities(nsamples * nsamples);
    const float nhinv = 1. / nh;
    start = std::chrono::high_resolution_clock::now();
    OMP_PFOR
    for(size_t i = 0; i < nsamples; ++i) {
        const Sig *const lhdat = hashes[i].data();
        for(size_t j = 0; j < nsamples; ++j)
            wsimilarities[i * nsamples + j] = nhinv * shared_rems(lhdat, hashes[j].data(), nh);
    }
    stop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Comparing took %g ms\n", std::chrono::duration<double, std::milli>(stop - start).count());
    std::FILE *matofp = std::fopen("scshriva.mat.txt", "w");
    if(!matofp) throw 1;
    for(size_t i = 0; i < nsamples; ++i) {
        for(size_t j = 0; j < nsamples; ++j) {
            std::fprintf(matofp, "%g", wsimilarities[i * nsamples + j]);
            std::fputc(j == nsamples - 1 ? '\n': '\t', matofp);
        }
    }
    std::fclose(matofp);
}
