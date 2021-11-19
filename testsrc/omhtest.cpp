#include "sketch/bmh.h"

int main (int argc, char **argv) {
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
}
