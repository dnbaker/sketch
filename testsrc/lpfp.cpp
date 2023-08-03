#include "sketch/lpcqf.h"
#include <chrono>

template<typename T, typename FT=double>
FT timelen(T x, T y) {
    return std::chrono::duration<FT, std::milli>(y - x).count();
}

template<typename FT, size_t SIGBITS = sizeof(FT) * 4>
int submain(size_t NITEMS) {
    auto id2w = [&](size_t i) {
        return 2;
    };
    size_t nentered = NITEMS * .75;
    size_t ss = NITEMS;
    sketch::LPCQF<FT, SIGBITS, sketch::IS_POW2> lpf(ss), lpf2(ss);
    std::vector<uint64_t> bulk;
    std::vector<size_t> counts;
    lpf += lpf2;
    auto ts = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < nentered; ++i) {
        lpf.update(nentered - i - 1, id2w(i));
    }
    auto ts2 = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "construction of %zu items took %gms, for %g million per minute\n", nentered, timelen(ts, ts2), nentered / timelen(ts, ts2) / 1e6 * 60.);
    for(size_t i = 0; i < nentered; ++i) {
        size_t inserted_key = nentered - i - 1;
        counts.push_back(id2w(i));
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
    std::fprintf(stderr, "ip: %0.10g\n", ip);
    lpf.batch_update(bulk.data(), bulk.size());
    const double new_ip = lpf.inner_product(lpf);
    std::fprintf(stderr, "ip: %0.10g. Difference: %0.6g\n", new_ip, new_ip - ip);
    return 0;
}

int main() {
    int ret = 0;
    for(const auto N: {size_t(1<<16), size_t(1) << 20}) {
    ret |= submain<float>(N)
        || submain<float, 0>(N)
        || submain<double>(N)
        || submain<double, 0>(N);
        if(ret) return ret;
    }
    return 0;
}
