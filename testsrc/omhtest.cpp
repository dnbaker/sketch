#include "sketch/bmh.h"

int main (int argc, char **argv) {
    size_t m = argc == 1 ? 1000: std::atoi(argv[1]);
    size_t l = argc <= 2 ? 10: std::atoi(argv[2]);
    sketch::omh::OMHasher<double> h(m, l);
    std::vector<char> s(1000, 0), s2(1000);
    for(size_t i = 0; i < 1000; ++i) s2[i] = i & 63 ? 0 : 1;
    auto lhh = h.hash(s.data(), s2.size());
    auto rhh = h.hash(s2.data(), s2.size());
    size_t n = 0;
    for(size_t i = 0; i < lhh.size(); ++i)
        n += lhh[i] == rhh[i];
    std::fprintf(stderr, "Shared hash regs: %zu/%zu\n", n, lhh.size());
}
