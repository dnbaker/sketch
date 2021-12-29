#include "sketch/lpcqf.h"
#include <chrono>

template<typename T>
double timelen(T x, T y) {
    return std::chrono::duration<double, std::milli>(y - x).count();
}

template<typename FT>
int submain(size_t NITEMS) {
    size_t nentered = NITEMS * .75;
    size_t ss = NITEMS;
    sketch::LPCQF<FT, sizeof(FT) * 4, sketch::IS_POW2> lpf(ss);
    std::vector<uint64_t> bulk;
    std::vector<size_t> counts;
    auto ts = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < nentered; ++i) {
        lpf.update(nentered - i - 1, i + 1);
    }
    auto ts2 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "construction of %zu items took %gms, for %g million per minute\n", nentered, timelen(ts, ts2), nentered / timelen(ts, ts2) / 1e6 * 60.);
    for(size_t i = 0; i < nentered; ++i) {
        size_t inserted_key = nentered - i - 1;
        counts.push_back(i + 1);
        bulk.push_back(inserted_key);
    }
    auto ts3 = std::chrono::high_resolution_clock::now();
    lpf.batch_update(bulk.data(), bulk.size(), counts.data());
    auto ts4 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "group construction of %zu items took %gms, for %g million per minute\n", nentered, timelen(ts3, ts4), nentered / timelen(ts3, ts4) / 1e6 * 60.);
    ts3 = std::chrono::high_resolution_clock::now();
    static constexpr size_t batchsize = 2048;
    for(size_t i = 0; i < (bulk.size() + batchsize - 1) / batchsize; ++i) {
        const size_t batchstart = i * batchsize;
        const size_t batchend = std::min(batchstart + batchsize, bulk.size());
        lpf.batch_update(&bulk[batchstart], batchend - batchstart, &counts[batchstart]);
    }
    lpf.batch_update(bulk.data(), bulk.size(), counts.data());
    ts4 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "batched batch construction of %zu items took %gms, for %g million per minute\n", nentered, timelen(ts3, ts4), nentered / timelen(ts3, ts4) / 1e6 * 60.);
    auto ip = lpf.inner_product(lpf);
    std::fprintf(stderr, "ip: %g\n", ip);
    lpf.batch_update(bulk.data(), bulk.size());
    ip = lpf.inner_product(lpf);
    std::fprintf(stderr, "ip: %g\n", ip);
    return 0;
}

int main() {
    int ret;
    for(const auto N: {size_t(1<<16), size_t(1) << 20, size_t(16) << 20}) {
    ret |= submain<float>(N)
        || submain<double>(N);
        if(ret) return ret;
    }
    return 0;
}
