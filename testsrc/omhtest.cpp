#include "sketch/bmh.h"

int main (int argc, char **argv) {
    size_t m = argc == 1 ? 1000: std::atoi(argv[1]);
    size_t l = argc <= 2 ? 10: std::atoi(argv[2]);
    size_t slen = argc <= 3 ? 1000: std::atoi(argv[3]);
    sketch::omh::OMHasher<double> h(m, l);
    std::vector<char> s(slen, 0), s2(slen);
    for(size_t i = 0; i < slen; ++i) s2[i] = i % 64 == 0;
    auto lhh = h.hash(s.data(), s2.size());
    auto rhh = h.hash(s2.data(), s2.size());
    size_t n = 0;
    for(size_t i = 0; i < lhh.size(); ++i)
        n += lhh[i] == rhh[i];
    std::fprintf(stderr, "Shared hash regs: %zu/%zu\n", n, lhh.size());
}
