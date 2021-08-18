#include "sketch/count_eq.h"
#include <iostream>

template<typename T>
int do_main() {
    std::vector<T, sketch::Allocator<T>> rhs(100, 0), lhs(100, 0);
    std::iota(rhs.begin(), rhs.end(), 0u);
    lhs = rhs;
    std::fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    lhs[17] = 1;
    assert(rhs[17] > lhs[17]);
    auto nm = sketch::eq::count_eq(lhs.data(), rhs.data(), 100);
    auto gtlti = sketch::eq::count_gtlt(lhs.data(), rhs.data(), 100);
    std::pair<size_t, size_t> gtlt = {gtlti.first, gtlti.second};
    assert(gtlt.first == 0);
    assert(gtlt.second == 1);
    std::fprintf(stderr, "nmatched: %zu\n", nm);
    for(size_t i = 0; i < 100; ++i) lhs[i] = rhs[i] + 1;
    gtlti = sketch::eq::count_gtlt(lhs.data(), rhs.data(), 100);
    gtlt = {gtlti.first, gtlti.second};
    std::fprintf(stderr, "[%s] gtlt %zu/%zu\n", __PRETTY_FUNCTION__, gtlt.first, gtlt.second);
    assert(gtlt.first == 100);
    lhs = rhs;
    if(sizeof(T) == 1) {
        std::memset(&rhs[0], 0, 100);
        std::memset(&lhs[0], 0, 100);
        auto nm = sketch::eq::count_eq_nibbles((const uint8_t *)lhs.data(), (const uint8_t *)rhs.data(), 200);
        std::fprintf(stderr, "nm: %zu\n", nm);
        assert(nm == 200);
    }
    double tt = 0., t2 = 0.;
    const size_t n = 1000000;
    wy::WyRand<uint64_t> rng(13);
    rhs.resize(n);
    lhs.resize(n);
    for(auto &i: rhs) i = rng();
    for(auto &i: lhs) i = rng();
    size_t nm2 = 0, nmb = 0;
    constexpr size_t NPER = sizeof(T) / sizeof(char);
    for(size_t i = 0; i < 1; ++i) {
        auto t = std::chrono::high_resolution_clock::now();
        nm2 += sketch::eq::count_eq_nibbles((const uint8_t *)lhs.data(), (const uint8_t *)rhs.data(), rhs.size() * NPER * 2);
        auto e = std::chrono::high_resolution_clock::now();
        tt += std::chrono::duration<double, std::milli>(e - t).count();
        t = std::chrono::high_resolution_clock::now();
        nmb += sketch::eq::count_eq_bytes((const uint8_t *)lhs.data(), (const uint8_t *)rhs.data(), rhs.size() * NPER);
        e = std::chrono::high_resolution_clock::now();
        t2 += std::chrono::duration<double, std::milli>(e - t).count();
    }
    std::fprintf(stderr, "%gms per %zu nibbles, %zu (%%%g) match\n", tt, rhs.size() * NPER * 2, nm2, double(nm2) / n / NPER / 2 * 100.);
    std::fprintf(stderr, "%gms per %zu bytes, %zu (%%%g) match\n", t2, rhs.size() * NPER, nmb, double(nmb) / n / NPER * 100.);
    if(nm != 99) {
        std::fprintf(stderr, "FAILURE!!!\n");
        return 1;
    }
    return 0;
}

int main() {
    int rc = do_main<uint16_t>() | do_main<uint32_t>() | do_main<uint64_t>() | do_main<uint8_t>();
    std::vector<char> lhs(1000), rhs(1000);
    for(size_t i = 0; i < 1000; ++i) {
        lhs[i] = 3 | (5 << 4); rhs[i] = 5 | (3 << 4);
        if((i & 7) == 7) {
            lhs[i] = 3 | (4 << 4); rhs[i] = 4 | (4 << 4);
        }
    }
    auto res = sketch::eq::count_gtlt_nibbles(lhs.data(), rhs.data(), 2000);
    std::fprintf(stderr, "Match: %zu/%zu/%zu\n", size_t(res.first), size_t(res.second), size_t(2000 - (res.first + res.second)));
    assert(res.first + res.second <= 2000u);
    return rc;
}
