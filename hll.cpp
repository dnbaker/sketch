#include "hll.h"
#include <stdexcept>
#include <cstring>
namespace hll {

void hll_t::sum() {
    sum_ = 0;
    for(unsigned i(0); i < m_; ++i) sum_ += 1. / (1ull << core_[i]);
    is_calculated_ = 1;
    LOG_DEBUG("Summed! Is calculated: %i\n", is_calculated_);
}

double hll_t::creport() const {
    if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to report."
                                                 " Try the report() function.");
    const double ret(alpha_ * m_ * m_ / sum_);
    // Correct for small values
    if(ret < m_ * 2.5) {
        int t(0);
        for(unsigned i(0); i < m_; ++i) t += (core_[i] == 0);
        if(t) return m_ * std::log((double)(m_) / t);
    }
    return ret;
    // We don't correct for too large just yet, but we should soon.
}

double hll_t::report() {
    if(!is_calculated_) sum();
    assert(is_calculated_);
    return hll_t::creport();
}

double hll_t::est_err() {
    if(!is_calculated_) sum();
    return cest_err();
}

double hll_t::cest_err() const {
    if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to estimate."
                                                 " Try the report() function.");
    return relative_error_ * creport();
}

hll_t const &hll_t::operator+=(const hll_t &other) {
    if(other.np_ != np_)
        LOG_EXIT("np_ (%zu) != other.np_ (%zu)\n", np_, other.np_);
#if HAS_AVX_512
    unsigned i;
    __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
    const __m512i *oels(reinterpret_cast<const __m512i *>(other.core_.data()));
    for(i = 0; i < m_ >> 6; ++i) els[i] = _mm512_or_epi64(els[i], oels[i]);
    if(m_ < 64) for(;i < m_; ++i) core_[i] |= other.core_[i];
#elif __AVX2__
    unsigned i;
    __m256i *els(reinterpret_cast<__m256i *>(core_.data()));
    const __m256i *oels(reinterpret_cast<const __m256i *>(other.core_.data()));
    for(i = 0; i < m_ >> 5; ++i) els[i] = _mm256_or_si256(els[i], oels[i]);
    if(m_ < 32) for(;i < m_; ++i) core_[i] |= other.core_[i];
#elif __SSE2__
    unsigned i;
    __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
    const __m128i *oels(reinterpret_cast<const __m128i *>(other.core_.data()));
    for(i = 0; i < m_ >> 4; ++i) els[i] = _mm_or_si128(els[i], oels[i]);
    if(m_ < 16) for(; i < m_; ++i) core_[i] |= other.core_[i];
#else
    for(unsigned i(0); i < m_; ++i) core_[i] |= other.core_[i];
#endif
    return *this;
}
hll_t const &hll_t::operator&=(const hll_t &other) {
    if(other.np_ != np_)
        LOG_EXIT("np_ (%zu) != other.np_ (%zu)\n", np_, other.np_);
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
double operator^(hll_t &first, hll_t &other) {
    hll_t tmp(first);
    tmp += other;
    return 2 * tmp.report() - first.report() - other.report();
}

// Returns the size of the set intersection
double operator&(hll_t &first, hll_t &other) {
    hll_t tmp(first);
    tmp &= other;
    return tmp.report();
}

// Clears, allows reuse with different np.
void hll_t::resize(std::size_t new_size) {
    new_size = roundup64(new_size);
    LOG_DEBUG("Resizing to %zu, with np = %zu\n", new_size, (std::size_t)std::log2(new_size));
    clear();
    core_.resize(new_size);
    np_ = (std::size_t)std::log2(new_size);
    size_t newm(new_size);
    memcpy((void *)&m_, &newm, sizeof(m_));
    alpha_ = make_alpha(m_);
    relative_error_ = 1.03896 / std::sqrt(m_);
}

hll_t operator+(const hll_t &one, const hll_t &other) {
    if(other.get_np() != one.get_np())
        LOG_EXIT("np_ (%zu) != other.get_np() (%zu)\n", one.get_np(), other.get_np());
    hll_t ret(one);
    return ret += other;
}

void hll_t::clear() {
     std::fill(std::begin(core_), std::end(core_), 0u);
     sum_ = is_calculated_ = 0;
}

} // namespace hll
