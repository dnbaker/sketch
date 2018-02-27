#ifndef HLL_HEADER_ONLY
#  include "hll.h"
#endif

#include <stdexcept>
#include <cstring>
#include <numeric>
#include <thread>
#include <cinttypes>
#include <atomic>
#include "kthread.h"

namespace hll {


#if !NDEBUG
template<typename T>
std::string arrstr(T it, T it2) {
    std::string ret;
    for(auto i(it); i != it2; ++i) ret += "," + std::to_string(*i);
    return ret;
}
#endif


_STORAGE_ void hll_t::sum() {
    using detail::SIMDHolder;
    uint64_t counts[64]{0};
    SIMDHolder tmp, *p((SIMDHolder *)core_.data()), *pend((SIMDHolder *)&*core_.end());
    do {
        tmp = *p++;
        tmp.inc_counts(counts);
    } while(p < pend);
    value_ = detail::calculate_estimate(counts, use_ertl_, m(), np_, alpha());
    is_calculated_ = 1;
}

template<typename CoreType>
struct parsum_data_t {
    std::atomic<uint64_t> *counts_; // Array decayed to pointer.
    const CoreType               &core_;
    const uint64_t              l_;
    const uint64_t             pb_; // Per-batch
};

template<typename CoreType>
_STORAGE_ void parsum_helper(void *data_, long index, int tid) {
    using detail::SIMDHolder;
    parsum_data_t<CoreType> &data(*(parsum_data_t<CoreType> *)data_);
    uint64_t local_counts[64]{0};
    SIMDHolder tmp, *p((SIMDHolder *)&data.core_[index * data.pb_]),
                    *pend((SIMDHolder *)&data.core_[std::min(data.l_, (index+1) * data.pb_)]);
    do {
        tmp = *p++;
        tmp.inc_counts(local_counts);
    } while(p < pend);
    for(uint64_t i = 0; i < 64ull; ++i) data.counts_[i] += local_counts[i];
}

_STORAGE_ void hll_t::parsum(int nthreads, std::size_t pb) {
    if(nthreads < 0) nthreads = std::thread::hardware_concurrency();
    std::atomic<uint64_t> acounts[64];
    std::memset(acounts, 0, sizeof acounts);
    parsum_data_t<decltype(core_)> data{acounts, core_, m(), pb};
    const uint64_t nr(core_.size() / pb + (core_.size() % pb != 0));
    kt_for(nthreads, parsum_helper<decltype(core_)>, &data, nr);
    uint64_t counts[64];
    std::memcpy(counts, acounts, sizeof(counts));
    value_ = detail::calculate_estimate(counts, use_ertl_, m(), np_, alpha());
    is_calculated_ = 1;
}


_STORAGE_ double hll_t::creport() const {
    if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to report."
                                                 " Try the report() function.");
    return value_;
}

_STORAGE_ double hll_t::cest_err() const {
    if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to report.");
    return relative_error() * creport();
}

_STORAGE_ double hll_t::est_err() noexcept {
    if(!is_calculated_) sum();
    return cest_err();
}

_STORAGE_ hll_t &hll_t::operator+=(const hll_t &other) {
    if(other.np_ != np_) {
        char buf[256];
        sprintf(buf, "For operator +=: np_ (%u) != other.np_ (%u)\n", np_, other.np_);
        throw std::runtime_error(buf);
    }
    unsigned i;
#if HAS_AVX_512
    __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
    const __m512i *oels(reinterpret_cast<const __m512i *>(other.core_.data()));
    for(i = 0; i < m() >> 6; ++i) els[i] = _mm512_max_epu8(els[i], oels[i]);
    if(m() < 64) for(;i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#elif __AVX2__
    __m256i *els(reinterpret_cast<__m256i *>(core_.data()));
    const __m256i *oels(reinterpret_cast<const __m256i *>(other.core_.data()));
    for(i = 0; i < m() >> 5; ++i) els[i] = _mm256_max_epu8(els[i], oels[i]);
    if(m() < 32) for(;i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#elif __SSE2__
    __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
    const __m128i *oels(reinterpret_cast<const __m128i *>(other.core_.data()));
    for(i = 0; i < m() >> 4; ++i) els[i] = _mm_max_epu8(els[i], oels[i]);
    if(m() < 16) for(; i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#else
    for(i = 0; i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#endif
    not_ready();
    return *this;
}

_STORAGE_ hll_t &hll_t::operator&=(const hll_t &other) {
    std::fprintf(stderr, "Warning: This method doesn't work very well at all. For some reason. Do not trust.\n");
    if(other.np_ != np_) {
        char buf[256];
        sprintf(buf, "For operator &=: np_ (%u) != other.np_ (%u)\n", np_, other.np_);
        throw std::runtime_error(buf);
    }
    unsigned i;
#if HAS_AVX_512
    __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
    const __m512i *oels(reinterpret_cast<const __m512i *>(other.core_.data()));
    for(i = 0; i < m() >> 6; ++i) els[i] = _mm512_min_epu8(els[i], oels[i]);
    if(m() < 64) for(;i < m(); ++i) core_[i] = std::min(core_[i], other.core_[i]);
#elif __AVX2__
    __m256i *els(reinterpret_cast<__m256i *>(core_.data()));
    const __m256i *oels(reinterpret_cast<const __m256i *>(other.core_.data()));
    for(i = 0; i < m() >> 5; ++i) {
        els[i] = _mm256_min_epu8(els[i], oels[i]);
    }
    if(m() < 32) for(;i < m(); ++i) core_[i] = std::min(core_[i], other.core_[i]);
#elif __SSE2__
    __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
    const __m128i *oels(reinterpret_cast<const __m128i *>(other.core_.data()));
    for(i = 0; i < m() >> 4; ++i) els[i] = _mm_min_epu8(els[i], oels[i]);
    if(m() < 16) for(;i < m(); ++i) core_[i] = std::min(core_[i], other.core_[i]);
#else
    for(i = 0; i < m(); ++i) core_[i] = std::min(core_[i], other.core_[i]);
#endif
    not_ready();
    return *this;
}

// Returns the size of a symmetric set difference.
_STORAGE_ double operator^(hll_t &first, hll_t &other) {
    return 2*(hll_t(first) + other).report() - first.report() - other.report();
}

// Returns the set intersection
_STORAGE_ hll_t operator&(hll_t &first, hll_t &other) {
    hll_t tmp(first);
    tmp &= other;
    return tmp;
}

_STORAGE_ hll_t operator+(const hll_t &one, const hll_t &other) {
    if(other.p() != one.p())
        LOG_EXIT("p (%zu) != other.p (%zu)\n", one.p(), other.p());
    hll_t ret(one);
    ret += other;
    return ret;
}

_STORAGE_ double intersection_size(hll_t &first, hll_t &other) noexcept {
    if(!first.is_calculated_) first.sum();
    if(!other.is_calculated_) other.sum();
    return intersection_size((const hll_t &)first, (const hll_t &)other);
}

_STORAGE_ double jaccard_index(hll_t &first, hll_t &other) noexcept {
    if(!first.is_calculated_) first.sum();
    if(!other.is_calculated_) other.sum();
    return jaccard_index((const hll_t &)first, (const hll_t &)other);
}

_STORAGE_ double jaccard_index(const hll_t &first, const hll_t &other) {
    double i(intersection_size(first, other));
    i /= (first.creport() + other.creport() - i);
    return i;
}

// Clears, allows reuse with different np.
_STORAGE_ void hll_t::resize(std::size_t new_size) {
    new_size = roundup64(new_size);
    LOG_DEBUG("Resizing to %zu, with np = %zu\n", new_size, (std::size_t)std::log2(new_size));
    clear();
    core_.resize(new_size);
    np_ = (std::size_t)std::log2(new_size);
}

_STORAGE_ void hll_t::clear() {
     std::fill(std::begin(core_), std::end(core_), 0u);
     value_ = is_calculated_ = 0;
}

_STORAGE_ std::string hll_t::to_string() const {
    std::string params(std::string("p:") + std::to_string(np_) + ";");
    return (params + (is_calculated_ ? std::to_string(creport()) + ", +- " + std::to_string(cest_err())
                          : desc_string()));
}

_STORAGE_ std::string hll_t::desc_string() const {
    char buf[512];
    std::sprintf(buf, "Size: %u. nb: %llu. error: %lf. Is calculated: %s. value: %lf\n",
                 np_, static_cast<long long unsigned int>(m()), relative_error(), is_calculated_ ? "true": "false", value_);
    return buf;
}

_STORAGE_ void hll_t::free() {
    decltype(core_) tmp{};
    std::swap(core_, tmp);
}

_STORAGE_ void hll_t::write(const int fileno) {
    uint32_t bf[3]{is_calculated_, use_ertl_, nthreads_};
    ::write(fileno, bf, sizeof(bf));
    ::write(fileno, &np_, sizeof(np_));
    ::write(fileno, &value_, sizeof(value_));
    ::write(fileno, core_.data(), core_.size() * sizeof(core_[0]));
}

_STORAGE_ void hll_t::read(const int fileno) {
    uint32_t bf[3];
    ::read(fileno, bf, sizeof(bf));
    is_calculated_ = bf[0]; use_ertl_ = bf[1]; nthreads_ = bf[2];
    ::read(fileno, &np_, sizeof(np_));
    ::read(fileno, &value_, sizeof(value_));
    core_.resize(m());
    ::read(fileno, core_.data(), core_.size());
}


_STORAGE_ void hll_t::write(std::FILE *fp) {
#if _POSIX_VERSION
    write(fileno(fp));
#else
    static_assert(false, "Needs posix for now, will write non-posix version later.");
#endif
}

_STORAGE_ void hll_t::read(std::FILE *fp) {
#if _POSIX_VERSION
    read(fileno(fp));
#else
    static_assert(false, "Needs posix for now, will write non-posix version later.");
#endif
}

_STORAGE_ void hll_t::read(const char *path) {
    std::FILE *fp(std::fopen(path, "rb"));
    if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + path);
    read(fp);
    std::fclose(fp);
}
_STORAGE_ void hll_t::write(const char *path) {
    std::FILE *fp(std::fopen(path, "wb"));
    if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + path);
    write(fp);
    std::fclose(fp);
}




} // namespace hll
