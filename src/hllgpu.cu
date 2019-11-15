#include "sketch/hllgpu.h"
#include <cstdio>
#include "omp.h"

using namespace sketch;
int main(int argc, char *argv[]) {
    size_t n = argc == 1 ? 60: std::atoi(argv[1]);
    int p = argc <= 2 ? 14: std::atoi(argv[2]);
    omp_set_num_threads(24);
//__host__ std::vector<float> all_pairs(const uint8_t *p, unsigned l2, size_t nhlls) {
    std::vector<hll::hll_t> hlls;
    hlls.reserve(n);
    for(int i = 0; i < n; ++i) hlls.emplace_back(p);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < 10000000;  ++i) {
        if(i % 100000 == 0) std::fprintf(stderr, "tid %d with %zu\n", omp_get_thread_num(), i);
        if(i & 1) {
            for(auto &h: hlls) h.addh(i);
        } else if(i & 2) {
            for(int j = 0; j < n; j += 2)
                hlls[j].addh(i);
        } else {
            for(int j = i % 7; j < n; j = j + 3)
                hlls[j].addh(i);
        }
    }
    std::fprintf(stderr, "Finished making now copy\n");
    std::vector<float> cards;
    cards.reserve(hlls.size());
    int mgs, bs, gs;
    cudaError_t ce;
    if((ce = cudaOccupancyMaxPotentialBlockSize(&mgs, &bs, calc_sizes_large, 0, 0)))
        throw CudaError(ce, "Failed to infer best block size and so on.");
    gs = (hlls.size() * hlls.size() / 2 + (bs - 1)) / bs;
    std::fprintf(stderr, "mgs: %d. bs: %d. gs: %d\n", mgs, bs, gs);
    for(auto &h: hlls) cards.push_back(h.report());
    for(size_t i = 0; i < hlls.size(); std::fprintf(stderr, "size: %lf\n", hlls[i++].report()));
    std::vector<uint8_t> cd(n << p);
    for(size_t i = 0; i < n; ++i) {
        std::memcpy(cd.data() + (i << p), hlls[i].data(), size_t(1) << p);
    }
    uint8_t *ddata;
    if(cudaMalloc((void **)&ddata, (n << p))) throw std::runtime_error("Failed to allocate on device");
    if(cudaMemcpy(ddata, cd.data(), n << p, cudaMemcpyHostToDevice)) throw std::runtime_error("Failed to copy to device");
    std::fprintf(stderr, "Finish copy\n");
    size_t time;
    auto sizes = all_pairsu(ddata, p, n, time);
    auto s2 = std::vector<uint32_t>(sizes.size());
    auto t = hrc::now();
    for(auto i = 0u; i < hlls.size(); ++i) {
        OMP_PRAGMA("omp parallel for")
        for(auto j = i + 1; j < hlls.size(); ++j) {
            s2[ij2ind(i, j, hlls.size())] = jaccard_index(hlls[i], hlls[j]);
        }
    }
    auto t2 = hrc::now();
    size_t time2 = (t2 - t).count();
    std::fprintf(stderr, "time diff: %zu\n", time2);
    std::fprintf(stderr, "time ratio: %lf\n", double(time2) / time);
    cudaFree(ddata);
}
