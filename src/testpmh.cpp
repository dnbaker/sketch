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
}
