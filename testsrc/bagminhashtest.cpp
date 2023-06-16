//#define STEP_COUNT
#define NO_BLAZE
#include "bmh.h"
#include <cassert>
#include <cinttypes>
#include <chrono>

using namespace sketch::wmh;
int main(int argc, char **argv) {
    if(argc > 3) {
        std::fprintf(stderr, "usage: %s <optional: n, # items> <optional: m, # sigs>\n", argv[0]);
        std::exit(EXIT_FAILURE);
    }
    size_t n = argc < 2 ? 1000: std::atoi(argv[1]);
    size_t m = argc < 3 ? 100: std::atoi(argv[2]);
    bmh_t<> bmh(m, true, true), bmh2(m, true, true), bmh3(m, true, true), bmh4(m, true, true), bmh7(m, true, true);
    pmh1_t<> pmh1(m), pmh2(m);
    pmh1_t<> pmh3(m), pmh4(m);
    pmh2_t<> pm21(m), pm22(m), pm23(m), pm24(m);
    pmh2_t<long double> lpm21(m), lpm22(m), lpm23(m), lpm24(m);
    pmh3_t<> pm31(m), pm32(m), pm33(m), pm34(m);
    assert(bmh.m() == m);
    assert(bmh2.m() == m);
    assert(std::equal(bmh.hvals_.data(), bmh.hvals_.data() + m, bmh2.hvals_.data()));
    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < n; ++i) {
        if(i % 16 == 0) bmh7.update_1(i, 1.767);
        bmh.update_1(i, 1.767);
        bmh2.update_1(i, 1.767);
        bmh3.update_1(i, 1.767);
        bmh4.update_2(i, 1.767);
        //if(i % 100 == 0) std::fprintf(stderr, "Processed %zu/%zu\r\n", i, n);
    }
    for(size_t i = 0; i < bmh.size(); ++i) {
        assert(bmh.idcounts_[i] == 1.767);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Updates for 5 BMH: %gs\n", static_cast<double>((stop - start).count() / 1000) * .001);
    start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < n; ++i) {
        pm21.update(i, 1.);
        pm22.update(i, 1);
        pm23.update(i, 1);
        pm24.update(i, 1);
        if(i % 8 == 0) pm24.update(i * i * i * i + 1710, 8);
    }
    stop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Updates for 5 PMH2: %gs\n", static_cast<double>((stop - start).count() / 1000) * .001);
    start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < n; ++i) {
        pm31.update(i, 1.);
        pm32.update(i, 1);
        pm33.update(i, 1);
        pm34.update(i, 1);
        if(i % 8 == 0) pm34.update(i * i * i * i + 1710, 8);
    }
    stop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Updates for 5 PMH3: %gs\n", static_cast<double>((stop - start).count() / 1000) * .001);
    start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < n; ++i) {
        pmh1.update(i, 1);
        pmh2.update(i, 1);
        pmh3.update(i, 1);
        pmh4.update(i, 1);
        if(i % 16 == 0) pmh3.update(i * i * 37 - i * 14, 16);
        //if(i % 100 == 0) std::fprintf(stderr, "Processed %zu/%zu\r\n", i, n);
    }
    stop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Updates for 8PMH: %gs\n", static_cast<double>((stop - start).count() / 1000) * .001);
#if ENABLE_SLEEF
    start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < n; ++i) {
        sph.update(i, 1);
        sph2.update(i * i, 2);
        sph2.update(i, 2);
        sph.update(i * i, 1);
        if(i % 16 == 0) sph3.update(i * i + 13, 16);
    }
    stop = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Updates for 4SPH: %gs\n", static_cast<double>((stop - start).count() / 1000) * .001);
#endif
    bmh4.finalize_2();
    auto s1 = bmh.to_sigs(), s2 = bmh2.to_sigs();
    auto s4 = bmh4.to_sigs();
    auto s7 = bmh7.to_sigs();
    auto ss1 = pmh1.to_sigs();
    auto ss2 = pmh2.to_sigs();
    pmh3.finalize();
    pmh4.finalize();
    auto ss3 = pmh3.to_sigs(), ss4 = pmh4.to_sigs();
    auto sp11 = pm21.to_sigs(), sp12 = pm24.to_sigs();
    size_t nmatch = 0, npmatch = 0, n2match = 0;
    for(size_t i = 0; i < m; ++i) {
        nmatch += s7[i] == s1[i];
        npmatch += ss3[i] == ss4[i];
        n2match += sp11[i] == sp12[i];
        //std::fprintf(stderr, "%" PRIu64 ":%" PRIu64 ":%" PRIu64 "\n", s1[i], s2[i], s3[i]);
        //std::fprintf(stderr, "[%" PRIu64 ":%" PRIu64 ":%" PRIu64 "]\n", s4[i], s5[i], s6[i]);
    }
    std::fprintf(stderr, "Expected 1/16 signatures to match. Matching: %zu/%zu\n", nmatch, m);
    std::fprintf(stderr, "Expected somewhere around half of signatures to match. Matching: %zu/%zu\n", npmatch, m);
    std::fprintf(stderr, "Expected somewhere around half of PMH2 signatures to match. Matching: %zu/%zu\n", n2match, m);
    assert(std::equal(s1.begin(), s1.end(), s2.begin()));
    assert(std::equal(s1.begin(), s1.end(), s4.begin()));
#ifdef STEP_COUNT
    using psc_t = decltype(poisson_process_step_counter);
    using v_t = std::pair<typename std::remove_const_t<psc_t::value_type::first_type>, typename std::remove_const_t<psc_t::value_type::second_type>>;
    std::vector<v_t> vals(poisson_process_step_counter.size());
    {
        auto vit = vals.begin();
        for(const auto &o: poisson_process_step_counter) *vit++ = {o.first, o.second};
    }
    std::sort(vals.begin(), vals.end(), [](auto x, auto y) {return std::tie(x.second, x.first) > std::tie(y.second, y.first);});
    auto vsum = std::accumulate(vals.begin(), vals.end(), 0., [](double sum, auto x) {return sum + x.second;});
    std::fprintf(stderr, "%zu total unique iterations, with %g total, for an average of %0.12g\n", vals.size(), vsum, vsum / vals.size());
    for(size_t i = 0; i < std::min(vals.size(), size_t(10)); ++i) {
        std::fprintf(stderr, "Count %zu (%zu) is the %zuth most common with %g%% of the total (%zu)\n", i, vals[i].first, i + 1,  100. * vals[i].second / vsum, vals[i].second);

    }
#endif

    //assert(!std::equal(s1.begin(), s1.end(), s3.begin()));
}
