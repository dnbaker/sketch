#include "sketch/lpcqf.h"

int main() {
    size_t nentered = 6;
    size_t ss = 32;
    for(const auto v: {500, 25}) {
        sketch::LPCQF<uint16_t, 2, sketch::IS_APPROXINC | sketch::IS_QUADRATIC_PROBING | sketch::IS_POW2> lp(ss);
        for(size_t i = 0; i < nentered; ++i) {
            lp.update(i, v);
            std::fprintf(stderr, "Value %d estimated as count %Lg\n", v, (lp.count_estimate(i)));
            std::fprintf(stderr, "Inserted %zu items\n", i + 1);
        }
    }
    {
        sketch::LPCQF<uint16_t, 2, sketch::IS_APPROXINC | sketch::IS_QUADRATIC_PROBING | sketch::IS_POW2, 52, 50> lp(ss);
        for(size_t i = 0; i < nentered; ++i) {
            lp.update(i, 25);
            std::fprintf(stderr, "Value 25 with base %Lg estimated as count %Lg\n", lp.approxlogb, (lp.count_estimate(i)));
            std::fprintf(stderr, "Inserted %zu items\n", i + 1);
        }
    }
    {
        sketch::LPCQF<uint16_t, 2, sketch::IS_APPROXINC | sketch::IS_QUADRATIC_PROBING | sketch::IS_POW2, 10003, 10000> lp(ss);
        for(size_t i = 0; i < nentered; ++i) {
            lp.update(i, 25);
            std::fprintf(stderr, "Value 25 with base %Lg estimated as count %Lg\n", lp.approxlogb, (lp.count_estimate(i)));
            std::fprintf(stderr, "Inserted %zu items\n", i + 1);
        }
    }
#if 0
    {
        sketch::LPCQF<uint16_t, 2, sketch::IS_APPROXINC | sketch::IS_QUADRATIC_PROBING | sketch::IS_POW2, 101, 3> lp(ss);
        for(size_t i = 0; i < nentered; ++i) {
            lp.update(i, 25);
            std::fprintf(stderr, "Value 25 with base %Lg estimated as count %Lg\n", lp.approxlogb, (lp.count_estimate(i)));
            std::fprintf(stderr, "Inserted %zu items\n", i + 1);
        }
    }
#endif
}
