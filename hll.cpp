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

namespace detail {
    static constexpr double LARGE_RANGE_CORRECTION_THRESHOLD = (1ull << 32) / 30.;
    static constexpr long double TWO_POW_32 = (1ull << 32) * 1.;
    static double small_range_correction_threshold(std::uint64_t m) {return 2.5 * m;}
}
using std::isnan;

static inline double calculate_estimate(std::uint64_t *counts,
                                        bool use_ertl, std::uint64_t m, std::uint32_t p, double alpha) {
    double sum = 0, value;
    for(unsigned i(0); i < 64; ++i) sum += counts[i] * (1. / (1ull << i));
    if(use_ertl) {
#if 0
        std::fprintf(stderr, "Calculating tau with m = %zu and count = %zu, p = %u\n",
                     size_t(m), size_t(counts[64 - p + 1]), p);
#endif
        double z = m * detail::gen_tau(static_cast<double>((m-counts[64 - p +1]))/(double)m);
        for(unsigned k = 64-p; k; --k) {
            z += counts[k];
            z *= 0.5;
        }
        z += m * detail::gen_sigma(static_cast<double>(counts[0])/static_cast<double>(m));
        return (m/(2.*std::log(2)))*m / z;
    } /* else */ 
    // Small/large range corrections
    // See Flajolet, et al. HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm
    if((value = (alpha * m * m / sum)) < detail::small_range_correction_threshold(m)) {
        if(counts[0]) {
            LOG_DEBUG("Small value correction. Original estimate %lf. New estimate %lf.\n",
                       value, m * std::log((double)m / counts[0]));
            value = m * std::log((double)(m) / counts[0]);
        }
    } else if(value > detail::LARGE_RANGE_CORRECTION_THRESHOLD) {
        const long double corr(-detail::TWO_POW_32 * std::log(1. - value / detail::TWO_POW_32));
        if(!isnan(corr)) value = corr;
        LOG_DEBUG("Large range correction returned nan. Defaulting to regular calculation.\n");
    }
    return value;
}

#if !NDEBUG
template<typename T>
std::string arrstr(T it, T it2) {
    std::string ret;
    for(auto i(it); i != it2; ++i) ret += "," + std::to_string(*i);
    return ret;
}
#endif


_STORAGE_ void hll_t::sum() {
    std::uint64_t counts[65]{0};
    for(const auto i: core_) ++counts[i];
    // Think about making a table of size 4096 and looking up two values at a time.
    value_ = calculate_estimate(counts, use_ertl_, m(), np_, alpha());
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
    std::uint64_t local_counts[65]{0};
    for(std::uint64_t i(index * data.pb_), e(std::min(data.l_, i + data.pb_)); i < e; ++i)
        ++local_counts[data.core_[i]];
    for(std::uint64_t i = 0; i < 65ull; ++i) data.counts_[i] += local_counts[i];
}

_STORAGE_ void hll_t::parsum(int nthreads, std::size_t pb) {
    if(nthreads < 0) nthreads = std::thread::hardware_concurrency();
    std::atomic<std::uint64_t> acounts[65];
    std::memset(acounts, 0, sizeof acounts);
    parsum_data_t<decltype(core_)> data{acounts, core_, m(), pb};
    const std::uint64_t nr(core_.size() / pb + (core_.size() % pb != 0));
    kt_for(nthreads, parsum_helper<decltype(core_)>, &data, nr);
    std::uint64_t counts[65];
    std::memcpy(counts, acounts, sizeof(counts));
    value_ = calculate_estimate(counts, use_ertl_, m(), np_, alpha());
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
    std::string params(std::string("p:") + std::to_string(np_) + ";");
    return (params + (is_calculated_ ? std::to_string(creport()) + ", +- " + std::to_string(cest_err())
                          : desc_string()));
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
    std::fprintf(stderr, "Is calc %u. ertl: %u. nthreads: %u\n", is_calculated_, use_ertl_, nthreads_);
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
    std::fprintf(stderr, "Is calc %u. ertl: %u. nthreads: %u\n", is_calculated_, use_ertl_, nthreads_);
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

_STORAGE_ double jaccard_index(hll_t &first, hll_t &other) noexcept {
    first.sum(); other.sum();
    return jaccard_index((const hll_t &)first, (const hll_t &)other);
}

_STORAGE_ double jaccard_index(const hll_t &first, const hll_t &other) {
    double i(intersection_size(first, other));
    i = i / (first.creport() + other.creport() - i);
    return i;
}

_STORAGE_ void dhll_t::sum() {
    std::uint64_t fcounts[65]{0};
    std::uint64_t rcounts[65]{0};
    const auto &core(hll_t::data());
    for(size_t i(0); i < core.size(); ++i) {
        ++fcounts[core[i]]; ++rcounts[dcore_[i]];
    }
    double forward_val = calculate_estimate(fcounts, use_ertl_, m(), np_, alpha());
    double reverse_val = calculate_estimate(rcounts, use_ertl_, m(), np_, alpha());
    value_ = (forward_val + reverse_val)*0.5;
    is_calculated_ = 1;
}


} // namespace hll
