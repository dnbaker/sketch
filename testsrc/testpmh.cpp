#include "pmh.h"

using namespace sketch;
using namespace jp;

int main() {
    PMinHasher<>hasher(10, 100);
    blaze::DynamicMatrix<float> zomg(1000, 10);
    std::vector<std::vector<uint16_t>> newvecs;
    randomize(zomg);
    for(size_t i = 0; i < zomg.rows(); ++i) {
        auto r = row(zomg, i);
        newvecs.push_back(hasher.template hash<decltype(r), std::vector<uint16_t>>(r));
    }
    auto func = [](auto ind, auto v) {/*std::fprintf(stderr, "ind: %zu. v: %lf\n", size_t(ind), double(v)); */};
    for_each_nonzero(row(zomg, 10), func);
    for_each_nonzero(newvecs.back(), func);
    blaze::CompressedVector<double> cv(49 * 49 + 1, 50);
    for(size_t i = 0; i < 50; ++i) {
        cv.insert(i * i, std::sqrt(i));
    }
    for_each_nonzero(cv, func);
}
