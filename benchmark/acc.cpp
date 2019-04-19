#include "sketch.h"

struct result {
    double mse_;
    double abse_;
    double bias_;
    result() {
        std::memset(this, 0, sizeof(*this));
    }
    void add(double v, double exact) {
        auto diff = (v - exact);
        mse_ += diff * diff;
        abse_ += std::abs(diff);
        bias_ += diff;
    }
    void div(double v) {mse_ /= v; abse_ /= v; bias_ /= v;}
};

int main() {
    constexpr size_t nelem = 1 << 20;
    size_t ntrials = 100;
    std::vector<uint64_t> v1, v2;
    v1.reserve(nelem);
    v2.reserve(nelem);
    std::fprintf(stderr, "#Type\tl2s\tmse\tme\tbias\n");
    for(size_t l2s = 9; l2s < 12; ++l2s) {
        result hr, sr, br, sr8, br8, sr4, br4, sr2, br2, wr;
        for(size_t master_index = 0; master_index < ntrials; ++master_index) {
            sketch::hll_t hll1(l2s), hll2(l2s);
            sketch::SuperMinHash<> smh1(1<<(l2s - 1), 16), smh2(1<<(l2s - 1), 16);
            sketch::BBitMinHasher<uint64_t> bmh1((l2s - 1), 16), bmh2((l2s - 1), 16);
            sketch::SuperMinHash<> smh3(1<<l2s, 8), smh4(1<<l2s, 8);
            sketch::BBitMinHasher<uint64_t> bmh3(l2s, 8), bmh4(l2s, 8);
            sketch::SuperMinHash<> smh5(1<<(l2s + 1), 4), smh6(1<<(l2s + 1), 4);
            sketch::BBitMinHasher<uint64_t> bmh5((l2s + 1), 4), bmh6((l2s + 1), 4);
            sketch::SuperMinHash<> smh7(1<<(l2s + 2), 2), smh8(1<<(l2s + 2), 2);
            sketch::BBitMinHasher<uint64_t> bmh7((l2s + 2), 2), bmh8((l2s + 2), 2);
            v1.clear(); v2.clear();
            wy::WyHash<uint64_t> gen(1337 * master_index);
            for(size_t i = 0; i < nelem; ++i) {
                auto v = gen();
                v1.push_back(v); v2.push_back(v);
                v1.push_back(gen());
                v2.push_back(gen());
            }
            for(const auto v: v1) {
                hll1.addh(v);
                smh1.addh(v);
                bmh1.addh(v);
                bmh3.addh(v);
                smh3.addh(v);
                smh5.addh(v);
                bmh5.addh(v);
                smh7.addh(v);
                bmh7.addh(v);
            }
            for(const auto v: v2) {
                hll2.addh(v);
                smh2.addh(v);
                bmh2.addh(v);
                bmh4.addh(v);
                smh4.addh(v);
                smh6.addh(v);
                bmh6.addh(v);
                smh8.addh(v);
                bmh8.addh(v);
            }
            auto smf1 = smh1.finalize();
            auto smf2 = smh2.finalize();
            auto bmf1 = bmh1.finalize();
            auto bmf2 = bmh2.finalize();
            auto bmf3 = bmh3.finalize();
            auto bmf4 = bmh4.finalize();
            auto smf3 = smh3.finalize();
            auto smf4 = smh4.finalize();
            auto bmf5 = bmh5.finalize();
            auto bmf6 = bmh6.finalize();
            auto smf5 = smh5.finalize();
            auto smf6 = smh6.finalize();
            auto bmf7 = bmh7.finalize();
            auto bmf8 = bmh8.finalize();
            auto smf7 = smh7.finalize();
            auto smf8 = smh8.finalize();
            auto w1 = bmh1.make_whll(), w2 = bmh2.make_whll();
            hr.add(hll1.jaccard_index(hll2), 0.33333333333333);
            sr.add(smf1.jaccard_index(smf2), 0.33333333333333);
            br.add(bmf1.jaccard_index(bmf2), 0.33333333333333);
            br8.add(bmf3.jaccard_index(bmf4), 0.33333333333333);
            sr8.add(smf3.jaccard_index(smf4), 0.33333333333333);
            br4.add(bmf5.jaccard_index(bmf6), 0.33333333333333);
            sr4.add(smf5.jaccard_index(smf6), 0.33333333333333);
            br2.add(bmf7.jaccard_index(bmf8), 0.33333333333333);
            sr2.add(smf7.jaccard_index(smf8), 0.33333333333333);
            wr.add(w1.jaccard_index(w2), 0.33333333333333);
        }
        hr.div(ntrials);
        br.div(ntrials);
        sr.div(ntrials);
        sr8.div(ntrials);
        br8.div(ntrials);
        sr4.div(ntrials);
        br4.div(ntrials);
        sr2.div(ntrials);
        br2.div(ntrials);
        wr.div(ntrials);
        std::fprintf(stderr, "HLL\t%zu\t%lf\t%lf\t%lf\n", l2s, hr.mse_, hr.abse_, hr.bias_);
        std::fprintf(stderr, "WHLL\t%zu\t%lf\t%lf\t%lf\n", l2s, wr.mse_, wr.abse_, wr.bias_);
        std::fprintf(stderr, "BBMH16\t%zu\t%lf\t%lf\t%lf\n", l2s, br.mse_, br.abse_, br.bias_);
        std::fprintf(stderr, "SMH16\t%zu\t%lf\t%lf\t%lf\n", l2s, sr.mse_, sr.abse_, sr.bias_);
        std::fprintf(stderr, "BBMH8\t%zu\t%lf\t%lf\t%lf\n", l2s, br8.mse_, br8.abse_, br8.bias_);
        std::fprintf(stderr, "SMH8\t%zu\t%lf\t%lf\t%lf\n", l2s, sr8.mse_, sr8.abse_, sr8.bias_);
        std::fprintf(stderr, "BBMH4\t%zu\t%lf\t%lf\t%lf\n", l2s, br4.mse_, br4.abse_, br4.bias_);
        std::fprintf(stderr, "SMH4\t%zu\t%lf\t%lf\t%lf\n", l2s, sr4.mse_, sr4.abse_, sr4.bias_);
        std::fprintf(stderr, "BBMH2\t%zu\t%lf\t%lf\t%lf\n", l2s, br2.mse_, br2.abse_, br2.bias_);
        std::fprintf(stderr, "SMH2\t%zu\t%lf\t%lf\t%lf\n", l2s, sr2.mse_, sr2.abse_, sr2.bias_);
    }
}
