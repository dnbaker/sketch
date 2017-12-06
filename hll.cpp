#ifndef HLL_HEADER_ONLY
#  include "hll.h"
#endif

#include <stdexcept>
#include <cstring>
#include <thread>
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

_STORAGE_ void hll_t::sum() {
    std::uint64_t counts[64]{0};
    for(const auto i: core_) ++counts[i];
    sum_ = 0;
    for(unsigned i(0); i < 64; ++i) sum_ += counts[i] * (1. / (1ull << i));
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
    memset(counts, 0, sizeof counts);
    parsum_data_t<decltype(core_)> data{counts, core_, m_, pb};
    const std::uint64_t nr(core_.size() / pb + (core_.size() % pb != 0));
    kt_for(nthreads, parsum_helper<decltype(core_)>, &data, nr);
    sum_ = 0;
    for(unsigned i = 0; i < 64; ++i) sum_ += counts[i] * (1. / (1ull << i));
    is_calculated_ = 1;
}


_STORAGE_ double hll_t::creport() const {
    if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to report."
                                                 " Try the report() function.");
    const long double ret(alpha_ * m_ * m_ / sum_);
    // Small/large range corrections
    // See Flajolet, et al. HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm
    if(ret < small_range_correction_threshold()) {
        int t(0);
        for(const auto i: core_) t += i == 0;
        if(t) {
            LOG_DEBUG("Small value correction. Original estimate %lf. New estimate %lf.\n",
                      ret, m_ * std::log((double)m_ / t));
            return m_ * std::log((double)(m_) / t);
        }
    } else if(ret > LARGE_RANGE_CORRECTION_THRESHOLD) {
        const long double corr(-TWO_POW_32 * std::log(1. - ret / TWO_POW_32));
        if(!isnan(corr)) return corr;
        LOG_WARNING("Large range correction returned nan. Defaulting to regular calculation.\n");
    }
    return ret;
}

_STORAGE_ double hll_t::cest_err() const {
    if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to report.");
    return relative_error_ * creport();
}

_STORAGE_ double hll_t::est_err() noexcept {
    if(!is_calculated_) sum();
    return cest_err();
}

_STORAGE_ hll_t const &hll_t::operator+=(const hll_t &other) {
    if(other.np_ != np_) {
        char buf[256];
        sprintf(buf, "For operator +=: np_ (%zu) != other.np_ (%zu)\n", np_, other.np_);
        throw std::runtime_error(buf);
    }
    unsigned i;
#if HAS_AVX_512
    __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
    const __m512i *oels(reinterpret_cast<const __m512i *>(other.core_.data()));
    for(i = 0; i < m_ >> 6; ++i) els[i] = _mm512_max_epu8(els[i], oels[i]);
    if(m_ < 64) for(;i < m_; ++i) core_[i] = std::max(core_[i], other.core_[i]);
#elif __AVX2__
    __m256i *els(reinterpret_cast<__m256i *>(core_.data()));
    const __m256i *oels(reinterpret_cast<const __m256i *>(other.core_.data()));
    for(i = 0; i < m_ >> 5; ++i) els[i] = _mm256_max_epu8(els[i], oels[i]);
    if(m_ < 32) for(;i < m_; ++i) core_[i] = std::max(core_[i], other.core_[i]);
#elif __SSE2__
    __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
    const __m128i *oels(reinterpret_cast<const __m128i *>(other.core_.data()));
    for(i = 0; i < m_ >> 4; ++i) els[i] = _mm_max_epu8(els[i], oels[i]);
    if(m_ < 16) for(; i < m_; ++i) core_[i] = std::max(core_[i], other.core_[i]);
#else
    for(i = 0; i < m_; ++i) core_[i] = std::max(core_[i], other.core_[i]);
#endif
    return *this;
}

_STORAGE_ hll_t const &hll_t::operator&=(const hll_t &other) {
    if(other.np_ != np_) {
        char buf[256];
        sprintf(buf, "For operator &=: np_ (%zu) != other.np_ (%zu)\n", np_, other.np_);
        throw std::runtime_error(buf);
    }
    unsigned i;
#if HAS_AVX_512
    __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
    const __m512i *oels(reinterpret_cast<const __m512i *>(other.core_.data()));
    for(i = 0; i < m_ >> 6; ++i) els[i] = _mm512_min_epu8(els[i], oels[i]);
    if(m_ < 64) for(;i < m_; ++i) core_[i] = std::min(core_[i], other.core_[i]);
#elif __AVX2__
    __m256i *els(reinterpret_cast<__m256i *>(core_.data()));
    const __m256i *oels(reinterpret_cast<const __m256i *>(other.core_.data()));
    for(i = 0; i < m_ >> 5; ++i) els[i] = _mm256_min_epu8(els[i], oels[i]);
    if(m_ < 32) for(;i < m_; ++i) core_[i] = std::min(core_[i], other.core_[i]);
#elif __SSE2__
    __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
    const __m128i *oels(reinterpret_cast<const __m128i *>(other.core_.data()));
    for(i = 0; i < m_ >> 4; ++i) els[i] = _mm_min_epu8(els[i], oels[i]);
    if(m_ < 16) for(;i < m_; ++i) core_[i] = std::min(core_[i], other.core_[i]);
#else
    for(i = 0; i < m_; ++i) core_[i] = std::min(core_[i], other.core_[i]);
#endif
    return *this;
}

// Returns the size of a symmetric set difference.
_STORAGE_ double operator^(hll_t &first, hll_t &other) {
    return 2*(hll_t(first) + other).report() - first.report() - other.report();
}

// Returns the set intersection
_STORAGE_ hll_t operator&(hll_t &first, hll_t &other) {
    hll_t tmp(first);
    return tmp &= other;
}

_STORAGE_ hll_t operator+(const hll_t &one, const hll_t &other) {
    if(other.get_np() != one.get_np())
        LOG_EXIT("np_ (%zu) != other.get_np() (%zu)\n", one.get_np(), other.get_np());
    hll_t ret(one);
    return ret += other;
}

// Returns the size of the set intersection
_STORAGE_ double intersection_size(const hll_t &first, const hll_t &other) {
    hll_t tmp(first);
    tmp &= other;
    return tmp.report();
}

_STORAGE_ double intersection_size(hll_t &first, hll_t &other) noexcept {
    hll_t tmp(first);
    tmp &= other;
    return tmp.creport();
}

// Clears, allows reuse with different np.
_STORAGE_ void hll_t::resize(std::size_t new_size) {
    new_size = roundup64(new_size);
    LOG_DEBUG("Resizing to %zu, with np = %zu\n", new_size, (std::size_t)std::log2(new_size));
    clear();
    core_.resize(new_size);
    np_ = (std::size_t)std::log2(new_size);
    m_ = new_size;
    alpha_ = make_alpha(m_);
    relative_error_ = 1.03896 / std::sqrt(m_);
}

_STORAGE_ void hll_t::clear() {
     std::fill(std::begin(core_), std::end(core_), 0u);
     sum_ = is_calculated_ = 0;
}

_STORAGE_ std::string hll_t::to_string() const {
    return is_calculated_ ? std::to_string(creport()) + ", +- " + std::to_string(cest_err())
                          : desc_string();
}

_STORAGE_ std::string hll_t::desc_string() const {
    char buf[1024];
    std::sprintf(buf, "Size: %zu. nb: %zu. error: %lf. Is calculated: %s. sum: %lf\n",
                 np_, m_, relative_error_, is_calculated_ ? "true": "false", sum_);
    return buf;
}

_STORAGE_ void hll_t::free() {
    decltype(core_) tmp{};
    std::swap(core_, tmp);
}

_STORAGE_ double jaccard_index(hll_t &first, hll_t &other) noexcept {
    double is(intersection_size(first, other));
    return is / (first.report() + other.report() - is);
}

_STORAGE_ double jaccard_index(const hll_t &first, const hll_t &other) {
    double is(intersection_size(first, other));
    return is / (first.creport() + other.creport() - is);
}

} // namespace hll
