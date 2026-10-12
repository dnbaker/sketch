#include "sketch/bmh.h"

int main (int argc, char **argv) {
    (void)argc; (void)argv;
    size_t m = argc == 1 ? 50: std::atoi(argv[1]);
    size_t l = argc <= 2 ? 2: std::atoi(argv[2]);
    sketch::omh::OMHasher<double> h(m, l);
    std::vector<std::string> strings {
        "die", "again", "tomorrow", "forever", "today", "todie"
    };
    const size_t ns = strings.size();
    std::vector<uint64_t> sigs;
    for(const auto &s: strings) {
        const auto sig = h(s.data(), s.size());
        sigs.insert(sigs.end(), sig.begin(), sig.end());
    }
    for(size_t i = 0; i < ns; ++i) {
        for(size_t j = 0; j < ns; ++j) {
            const size_t shared = std::inner_product(&sigs[m * i], &sigs[(i + 1) * m], &sigs[j * m], size_t(0), std::plus<>(), std::equal_to<>());
            std::fprintf(stderr, "%s and %s share %zu OMH registers\n", strings[i].data(), strings[j].data(), shared);
        }
    }
    {   // finalize index-width regression: positions are input offsets and
        // must keep index width even when the element type is narrow
        // (char). n = 300 forces offsets >= 256 that a narrow index type
        // would truncate; the golden signature below is the exact-offset
        // value and FAILS against the truncating implementation.
        std::mt19937_64 rng(42);
        std::vector<char> d300(300);
        for(auto &c: d300) c = char(rng());
        sketch::omh::OMHasher<double> hg(8, 3);
        const auto sig = hg.hash(d300.data(), d300.size());
        const uint64_t expect[8] = {1415180217776780817ULL, 9348351191460450954ULL,
                                    16474715488300614191ULL, 4505046359807914828ULL,
                                    9015032555383530225ULL, 14441615962476949922ULL,
                                    2188117707881350514ULL, 9608247674039800497ULL};
        bool ok = sig.size() == 8;
        for(size_t i = 0; ok && i < 8; ++i) ok = sig[i] == expect[i];
        if(!ok) {
            std::fprintf(stderr, "OMH finalize index-width regression FAILED (got:");
            for(auto x: sig) std::fprintf(stderr, " %llu", (unsigned long long)x);
            std::fprintf(stderr, ")\n");
            return 1;
        }
        std::fprintf(stderr, "OMH finalize index-width regression passed\n");
    }
    return 0;
}
