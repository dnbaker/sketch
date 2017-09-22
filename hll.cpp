#include "hll.h"
#include <stdexcept>
#include <cstring>
#include <thread>
#include <atomic>
#include "kthread.h"
namespace hll {

constexpr long double TWO_POW_32 = (1ull << 32) * 1.;

void hll_t::sum() {
#if USE_OPENMP
    std::atomic<std::uint64_t> counts[64];
#else
    std::uint64_t counts[64];
#endif
    std::memset(counts, 0, sizeof counts);
#if USE_OPENMP
    // Do not use this. The atomic operations are very expensive.
    std::fprintf(stderr, "Using openmp\n");
    // TODO: avoid temporaries of
    #pragma omp parallel for schedule (static,16384)
    for(std::uint64_t i = 0; i < m_; ++i) {
        ++counts[core_[i]];
    }
#else
    for(const auto i: core_) ++counts[i];
#endif
    sum_ = 0;
    for(unsigned i(0); i < 64; ++i) sum_ += counts[i] * (1. / (1ull << i));
    is_calculated_ = 1;
}

template<typename CoreType>
struct parsum_data_t {
    std::atomic<std::uint64_t>      *counts_;
    const CoreType &core_;
    const std::uint64_t                 l_;
    const std::uint64_t                pb_; // Per-batch
    //parsum_data_t(std::atomic<std::uint64_t> *arr, const std::vector<std::uint8_t> &core, std::uint64_t m, std::uint64_t pb=-1):
    //        counts_(arr), core_(core), l_(m), pb_(pb == UINT64_C(-1) ? 1 << 16: pb) {}
};

template<typename CoreType>
void parsum_helper(void *data_, long index, int tid) {
    parsum_data_t<CoreType> &data(*(parsum_data_t<CoreType> *)data_);
    std::uint64_t local_counts[64]{0};
    for(std::uint64_t i(index * data.pb_), e(std::min(data.l_, i + data.pb_)); i < e; ++i)
        ++local_counts[data.core_[i]];
    for(std::uint64_t i(0); i < 64ull; ++i) data.counts_[i] += local_counts[i];
}

void hll_t::parsum(int nthreads, std::size_t pb) {
    if(nthreads < 0) nthreads = std::thread::hardware_concurrency();
    std::atomic<std::uint64_t> counts[64];
    memset(counts, 0, sizeof counts);
    parsum_data_t<decltype(core_)> data{counts, core_, m_, pb};
    std::uint64_t nr(core_.size() / pb + (core_.size() % pb != 0));
    kt_for(nthreads, parsum_helper<decltype(core_)>, &data, nr);
    sum_ = 0;
    for(unsigned i(0); i < 64; ++i) sum_ += counts[i] * (1. / (1ull << i));
    is_calculated_ = 1;
}


double hll_t::creport() const {
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
        if(!std::isnan(corr)) return corr;
        LOG_WARNING("Large range correction returned nan. Defaulting to regular calculation.\n");
    }
    return ret;
}

double hll_t::cest_err() const {
    if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to report.");
    return relative_error_ * creport();
}

double hll_t::est_err() noexcept {
    if(!is_calculated_) sum();
    return cest_err();
}

hll_t const &hll_t::operator+=(const hll_t &other) {
    if(other.np_ != np_) {
        char buf[256];
        sprintf(buf, "For operator +=: np_ (%zu) != other.np_ (%zu)\n", np_, other.np_);
        throw std::runtime_error(buf);
    }
    unsigned i;
#if HAS_AVX_512
    __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
    const __m512i *oels(reinterpret_cast<const __m512i *>(other.core_.data()));
    for(i = 0; i < m_ >> 6; ++i) els[i] = _mm512_or_epi64(els[i], oels[i]);
    if(m_ < 64) for(;i < m_; ++i) core_[i] |= other.core_[i];
#elif __AVX2__
    __m256i *els(reinterpret_cast<__m256i *>(core_.data()));
    const __m256i *oels(reinterpret_cast<const __m256i *>(other.core_.data()));
    for(i = 0; i < m_ >> 5; ++i) els[i] = _mm256_or_si256(els[i], oels[i]);
    if(m_ < 32) for(;i < m_; ++i) core_[i] |= other.core_[i];
#elif __SSE2__
    __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
    const __m128i *oels(reinterpret_cast<const __m128i *>(other.core_.data()));
    for(i = 0; i < m_ >> 4; ++i) els[i] = _mm_or_si128(els[i], oels[i]);
    if(m_ < 16) for(; i < m_; ++i) core_[i] |= other.core_[i];
#else
    for(i = 0; i < m_; ++i) core_[i] |= other.core_[i];
#endif
    return *this;
}

hll_t const &hll_t::operator&=(const hll_t &other) {
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
double operator^(hll_t &first, hll_t &other) {
    return 2*(hll_t(first) + other).report() - first.report() - other.report();
}

// Returns the set intersection
hll_t operator&(hll_t &first, hll_t &other) {
    hll_t tmp(first);
    return tmp &= other;
}

hll_t operator+(const hll_t &one, const hll_t &other) {
    if(other.get_np() != one.get_np())
        LOG_EXIT("np_ (%zu) != other.get_np() (%zu)\n", one.get_np(), other.get_np());
    hll_t ret(one);
    return ret += other;
}

// Returns the size of the set intersection
double intersection_size(const hll_t &first, const hll_t &other) {
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
    m_ = new_size;
    alpha_ = make_alpha(m_);
    relative_error_ = 1.03896 / std::sqrt(m_);
}

void hll_t::clear() {
     std::fill(std::begin(core_), std::end(core_), 0u);
     sum_ = is_calculated_ = 0;
}

std::string hll_t::to_string() const {
    return is_calculated_ ? std::to_string(creport()) + ", +- " + std::to_string(cest_err())
                          : desc_string();
}

std::string hll_t::desc_string() const {
    char buf[1024];
    std::sprintf(buf, "Size: %zu. nb: %zu. error: %lf. Is calculated: %s. sum: %lf\n",
                 np_, m_, relative_error_, is_calculated_ ? "true": "false", sum_);
    return buf;
}

void hll_t::free() {
    core_.resize(0);
    core_.shrink_to_fit();
}

} // namespace hll
