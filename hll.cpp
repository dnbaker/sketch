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

#ifdef HLL_HEADER_ONLY
#  define _STORAGE_ inline
#else
#  define _STORAGE_
#endif

namespace hll {
using std::isnan;

static constexpr long double TWO_POW_32 = (1ull << 32) * 1.;

// Small/large range corrections
// See Flajolet, et al. HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm
#define SUM_CORE \
    double sum = 0;\
    for(unsigned i(0); i < 64; ++i) sum += counts[i] * (1. / (1ull << i));\
    if(use_ertl_) {\
        double z = m() * detail::gen_tau(static_cast<double>(m()-counts[1 - np_ +1])/static_cast<double>(m()));\
        for(unsigned k = 1-np_; k; --k) {\
            z += counts[k];\
            z *= 0.5;\
        }\
        z += m() * detail::gen_sigma(static_cast<double>(counts[0])/static_cast<double>(m()));\
        value_ = (m()/(2.*std::log(2)))*m() / z;\
        return;\
    } /* else */ \
    if((value_ = (alpha() * m() * m() / sum)) < small_range_correction_threshold()) {\
        if(counts[0]) {\
            LOG_DEBUG("Small value correction. Original estimate %lf. New estimate %lf.\n",\
                       value_, m() * std::log((double)m() / counts[0]));\
            value_ = m() * std::log((double)(m()) / counts[0]);\
        }\
    } else if(value_ > LARGE_RANGE_CORRECTION_THRESHOLD) {\
        const long double corr(-TWO_POW_32 * std::log(1. - value_ / TWO_POW_32));\
        if(!isnan(corr)) value_ = corr;\
        LOG_DEBUG("Large range correction returned nan. Defaulting to regular calculation.\n");\
    }

_STORAGE_ void hll_t::sum() {
    std::uint64_t counts[64]{0};
    for(const auto i: core_) ++counts[i];
    SUM_CORE
    is_calculated_ = 1;
}

template<typename CoreType>
struct parsum_data_t {
    std::atomic<std::uint64_t> *counts_; // Array decayed to pointer.
    const CoreType               &core_;
    const std::uint64_t              l_;
    const std::uint64_t             pb_; // Per-batch
};

template<typename CoreType>
_STORAGE_ void parsum_helper(void *data_, long index, int tid) {
    parsum_data_t<CoreType> &data(*(parsum_data_t<CoreType> *)data_);
    std::uint64_t local_counts[64]{0};
    for(std::uint64_t i(index * data.pb_), e(std::min(data.l_, i + data.pb_)); i < e; ++i)
        ++local_counts[data.core_[i]];
    for(std::uint64_t i = 0; i < 64ull; ++i) data.counts_[i] += local_counts[i];
}

_STORAGE_ void hll_t::parsum(int nthreads, std::size_t pb) {
    if(nthreads < 0) nthreads = std::thread::hardware_concurrency();
    std::atomic<std::uint64_t> counts[64];
    std::memset(counts, 0, sizeof counts);
    parsum_data_t<decltype(core_)> data{counts, core_, m(), pb};
    const std::uint64_t nr(core_.size() / pb + (core_.size() % pb != 0));
    kt_for(nthreads, parsum_helper<decltype(core_)>, &data, nr);
    SUM_CORE
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

_STORAGE_ hll_t const &hll_t::operator+=(const hll_t &other) {
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

_STORAGE_ hll_t const &hll_t::operator&=(const hll_t &other) {
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
    if(other.get_np() != one.get_np())
        LOG_EXIT("np_ (%zu) != other.get_np() (%zu)\n", one.get_np(), other.get_np());
    hll_t ret(one);
    ret += other;
    return ret;
}

// Returns the size of the set intersection
_STORAGE_ double intersection_size(const hll_t &first, const hll_t &other) {
    return first.creport() + other.creport() - hll_t(first + other).report();
}

_STORAGE_ double intersection_size(hll_t &first, hll_t &other) noexcept {
    first.sum(); other.sum();
    return intersection_size((const hll_t &)first, (const hll_t &)other);
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
    return is_calculated_ ? std::to_string(creport()) + ", +- " + std::to_string(cest_err())
                          : desc_string();
}

_STORAGE_ std::string hll_t::desc_string() const {
    char buf[1024];
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
    uint8_t buf[sizeof(value_) + sizeof(np_) + sizeof(bf)];
    uint8_t *ptr(buf);
    std::memcpy(ptr, &np_, sizeof(np_));
    ptr += sizeof(np_);
    std::memcpy(ptr, &value_, sizeof(value_));
    ptr += sizeof(value_);
    std::memcpy(ptr, bf, sizeof(bf));
    ptr += sizeof(bf);
    ::write(fileno, ptr, sizeof(buf));
    ::write(fileno, core_.data(), core_.size());
}

_STORAGE_ void hll_t::read(const int fileno) {
    uint8_t buf[sizeof(double) + sizeof(int) + sizeof(uint32_t)];
    ::read(fileno, buf, sizeof(buf));
    uint8_t *ptr(buf);
    std::memcpy(&np_, ptr, sizeof(np_));
    ptr += sizeof(np_);
    std::memcpy(&value_, ptr, sizeof(value_));
    ptr += sizeof(value_);
    uint32_t bf[3];
    std::memcpy(bf, ptr, sizeof(bf));
    ptr += sizeof(bf);
    is_calculated_ = bf[0]; use_ertl_ = bf[1]; nthreads_ = bf[2];
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

_STORAGE_ double jaccard_index(hll_t &first, hll_t &other) noexcept {
    first.sum(); other.sum();
    return jaccard_index((const hll_t &)first, (const hll_t &)other);
}

_STORAGE_ double jaccard_index(const hll_t &first, const hll_t &other) {
    double i(intersection_size(first, other));
    i = i / (first.creport() + other.creport() - i);
    return i;
}


} // namespace hll
