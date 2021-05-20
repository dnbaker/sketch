#ifndef FAST_APPROX_LOG_FLOG_H__
#define FAST_APPROX_LOG_FLOG_H__

namespace sketch {

namespace fastlog {
    static inline long double flog(long double x) {
        __uint128_t yi;
        std::memcpy(&yi, &x, sizeof(x));
        return yi * 3.7575583950764744255e-20L - 11356.176832703863597L;
    }
    static inline double flog(double x) {
        uint64_t yi;
        std::memcpy(&yi, &x, sizeof(yi));
        return yi * 1.539095918623324e-16 - 709.0895657128241;
    }
    static inline float flog(float x) {
        uint32_t yi;
        std::memcpy(&yi, &x, sizeof(yi));
        return yi * 8.2629582881927490e-8f - 88.02969186f;
    }
}

}

#endif
