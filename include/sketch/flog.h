#ifndef FAST_APPROX_LOG_FLOG_H__
#define FAST_APPROX_LOG_FLOG_H__

namespace sketch {

namespace fastlog {
    static inline long double flog(long double x) {
        static constexpr long double mul = 3.7575583950764744255e-20L;
        static constexpr long double offset = -11356.176832703863597L;
        __uint128_t yi;
        std::memcpy(&yi, &x, sizeof(x));
        return std::fma(static_cast<long double>(yi), mul, offset);
    }
    static inline double flog(double x) {
        static constexpr double mul = 1.539095918623324e-16;
        static constexpr double offset = -709.0895657128241;
        uint64_t yi;
        std::memcpy(&yi, &x, sizeof(yi));
        return std::fma(static_cast<double>(yi), mul, offset);
    }
    static inline float flog(float x) {
        static constexpr float mul = 8.2629582881927490e-8f;
        static constexpr float offset = -88.02969186f;

        uint32_t yi;
        std::memcpy(&yi, &x, sizeof(yi));
        return std::fma(static_cast<float>(yi), mul, offset);
    }
}

}

#endif
