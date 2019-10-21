#include "sparse.h"
#include <iostream>
#include <map>
using namespace sketch;
using namespace hll;
using namespace sparse;
#if !NDEBUG
#else
static_assert(false, "WOO");
#endif
class Timer {
    using TpType = std::chrono::system_clock::time_point;
    std::string name_;
    TpType start_, stop_;
public:
    Timer(std::string &&name=""): name_{std::move(name)}, start_(std::chrono::system_clock::now()) {}
    void stop() {stop_ = std::chrono::system_clock::now();}
    void restart() {start_ = std::chrono::system_clock::now();}
    double report() {return std::chrono::duration_cast<std::chrono::nanoseconds>(stop_ - start_).count(); std::cerr << "Took " << std::chrono::duration_cast<std::chrono::nanoseconds>(stop_ - start_).count() << "ns for task '" << name_ << "'\n";}
    ~Timer() {stop(); /* hammertime */ report();}
    void rename(const char *name) {name_ = name;}
};

int main() {
    enum {FIRST_LOOP_N = 150, HLL_SIZE=16};
    for(const auto size: {FIRST_LOOP_N * 1, FIRST_LOOP_N * 2, FIRST_LOOP_N * 512/* FIRST_LOOP_N << 10, FIRST_LOOP_N << 16 */}) {
        hll_t h1(HLL_SIZE), h2(HLL_SIZE);
        h1.set_jestim(ERTL_MLE);
        h2.set_jestim(ERTL_MLE);
        auto rshift = 64 - h1.p(), lshift = h1.p();
        std::map<uint32_t, uint8_t> tmp;
        std::vector<uint32_t> i1, i2;
        Timer zomg("");
        size_t zomgsum = 0, vtimesum = 0;
        for(size_t i = 0; i < size_t(size); ++i) {
            const auto v = h1.hash(i);
            auto ind = v >> rshift;
            auto val = uint8_t(clz(v << lshift) + 1);
            auto encval = SparseHLL32::encode_value(ind, val);
            assert(SparseHLL32::get_index(SparseHLL32::encode_value(ind, uint8_t(val))) == (ind));
            assert(SparseHLL32::get_value(SparseHLL32::encode_value(ind, uint8_t(val))) == uint8_t(val));
            zomg.restart();
            {
            auto it = tmp.find(ind);
            if(it == tmp.end()) it = tmp.emplace(ind, val).first;
            else          it->second = std::max(it->second, val);
            }
            zomg.stop();
            zomgsum += zomg.report();
            assert(ind == SparseHLL32::get_index(encval));
            assert((encval >> 6) == ind);
            zomg.restart();
            i1.push_back(encval);
            zomg.stop();
            vtimesum += zomg.report();
            h1.addh(i);
        }
        std::fprintf(stderr, "before div vtimesum is %zu, zomgsum is %zu\n", vtimesum, zomgsum);
        vtimesum /= size;
        zomgsum /= size;
        std::fprintf(stderr, "after div vtimesum is %zu, zomgsum is %zu\n", vtimesum, zomgsum);
        std::fprintf(stderr, "Runtime for updating tmp variable: %zu\n", size_t(zomgsum) / 150);
        for(size_t i = 140; i < 240; ++i) {
            const auto v = h2.hash(i);
            auto ind = v >> rshift;
            auto val = uint8_t(clz(v << lshift) + 1);
            auto encval = SparseHLL32::encode_value(ind, val);
            h2.addh(i);
            i2.push_back(encval);
        }
        h1.report();
        h2.report();
        SparseHLL<> h(h2), hh = h;
        SparseHLL<> h1fromv(i1), h2fromv(i2);
        assert(h2fromv == h);
        assert(h.is_sorted());
        assert(hh.is_sorted());
        std::fprintf(stderr, "h JI with h2: %.16lf\n", h.jaccard_index(h2));
        std::fprintf(stderr, "h JI with h1: %.16lf\n", h.jaccard_index(h1));
        std::fprintf(stderr, "h2 JI with h1: %.16lf. (IL size %zu)\n", h2.jaccard_index(h1), i1.size());
        std::vector<uint32_t> i1copy = i1;
        flatten(i1copy);
        double single_flattentime, single_maptime;
        std::array<double, 3> flattened_vals;
        {
            Timer t("zomg");
            flattened_vals = flatten_and_query(i1copy, h2, nullptr, true);
            t.stop();
            single_flattentime = t.report();
            t.restart();
            auto vals = pair_query(tmp, h1);
            t.stop();
            std::fprintf(stderr, "Single flatten and query, including sort time %zu\n", size_t(t.report()));
            std::fprintf(stderr, "vals: %f|%f|%f\n", vals[0], vals[1], vals[2]);
            std::fprintf(stderr, "vtime each: %zu. zomgtime: %zu. Bulk processing for vector %zu. Total score for vector approach: %zu. Total score for other approach: %zu. Query time for map: %zu\n",
                                  size_t(vtimesum), size_t(zomgsum), size_t(single_flattentime), vtimesum * FIRST_LOOP_N + size_t(single_flattentime), size_t(zomgsum * FIRST_LOOP_N + t.report()), size_t(t.report()));
            t.restart();
            auto ovals = h.jaccard_index(h2);
            std::fprintf(stderr, "ovals: %f\n", ovals);
            for(size_t i = 99; i--;ovals = h.jaccard_index(h2));
            t.stop();
            std::fprintf(stderr, "Single dense HLL query %zu\n", size_t(t.report()) / 100);
        }
        {
            Timer t("zomg");
            for(size_t i = 0; i < 100; ++i) {
                flattened_vals = flatten_and_query(i1copy, h2, nullptr, true);
            }
            t.stop();
            std::fprintf(stderr, "time for 100 i1copy parses: %zu\n", size_t(t.report()));
        }
        std::fprintf(stderr, "Flattened vals from i1: %lf/%lf/%f. ci: %lf. True ci: %lf\n", flattened_vals[0], flattened_vals[1], flattened_vals[2], flattened_vals[2] / (flattened_vals[0] + flattened_vals[2]), h1.containment_index(h2));
        i1copy = i2;
        flattened_vals = flatten_and_query(i2, h2);
        std::fprintf(stderr, "Flattened vals: %lf/%lf/%f. True: %lf\n", flattened_vals[0], flattened_vals[1], flattened_vals[2], h2.report());
        Timer t("zomg");
        auto vals = pair_query(tmp, h1);
        t.stop();
        std::fprintf(stderr, "Time for pair query with map: %zu\n", size_t(t.report()));
        single_maptime = t.report();
        std::fprintf(stderr, "map is %lf as fast as vector\n", double(single_flattentime) / double(single_maptime));
        auto it = tmp.begin();
        size_t ind = 0;
        i1copy = i1;
        flatten(i1copy);
        assert(tmp.size() == i1copy.size());
        assert(std::is_sorted(i1copy.begin(), i1copy.end()));
        i1copy = i1;
        flatten(i1copy);
        SparseHLL<> fil(i1copy);
        assert(fil == SparseHLL<>(h1));
        while(it != tmp.end() && ind < i1copy.size()) {
            //std::fprintf(stderr, "At ind %zu, Oval: %u. our val: %u. Expec %zu in tmp size and %zu in i1copy\n", ind, SparseHLL32::encode_value(it->first, it->second), i1copy[ind], tmp.size(), i1copy.size());
            assert(SparseHLL32::encode_value(it->first, it->second) == i1copy[ind]);
            ++it, ++ind;
        }
        std::fprintf(stderr, "iterator finished ? %s. index into i1copy finished? %s\n", it == tmp.end() ? "true": "false", ind == i1copy.size() ? "true": "NUH UH");
        std::fprintf(stderr, "Flattened vals: %lf/%lf/%fl\n", vals[0], vals[1], vals[2]);
        assert(vals[0] == 0. && vals[1] == 0.);
        hll_t uh = h1 + h2;
        uh.set_jestim(ERTL_MLE);
        uh.not_ready();
        uh.sum();
        auto us = h2.union_size(h1);
        assert(uh.report() == us);
        assert(h.jaccard_index(h2) == 1.);
        assert(h.jaccard_index(h1) == h2.jaccard_index(h1));
    }
}
