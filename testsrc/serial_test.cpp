#include "hll.h"
#include "bbmh.h"
#include "sketch/setsketch.h"
#include "aesctr/wy.h"
#include <chrono>
using namespace sketch;
using namespace hll;

inline auto gett() {return std::chrono::high_resolution_clock::now();}

int main(int argc, char *argv[]) {
    unsigned nelem(argc > 1 ? std::atoi(argv[1]): 1000000);
    {
        std::fprintf(stderr, "HLL subsection\n");
        hll_t h(18);

        auto start = gett();
        for(unsigned i(0); i < nelem; h.addh(++i));
        auto end = gett();
        std::fprintf(stderr, "%u insertions takes %lf us\n", nelem, std::chrono::nanoseconds(end - start).count() / 1000.);
        h.sum();
        h.not_ready();
        h.set_estim(hll::ORIGINAL);
        auto strform = h.to_string();
        std::fprintf(stderr, "h count: %lf. String: %s\n", h.report(), h.to_string().data());
        h.write("SaveSketch.hll");
        hll_t h2("SaveSketch.hll");
        assert(h2.get_is_ready() == h.get_is_ready());
        std::fprintf(stderr, "h2 count: %lf. String: %s\n", h2.report(), h2.to_string().data());
        auto strformh2 = h2.to_string();
        if(std::strcmp(h2.to_string().data(), h.to_string().data())) {
        //if(std::strcmp(strformh2.data(), strform.data())) {
            size_t i;
            for(i = 0; i < std::min(strform.size(), strformh2.size()); ++i)
                if(strformh2[i] != strform[i]) break;
            std::fprintf(stderr, "sizes: %zu, %zu. First diff: %zu. (%c/%c)\n", strform.size(), strformh2.size(), i, strform[i], strformh2[i]);
            throw std::runtime_error("Serialized form differs from original");
        }
        h2.sum();
        std::fprintf(stderr, "After resumming: h2 count: %lf. String: %s\n", h2.report(), h2.to_string().data());
    }
    {
        std::fprintf(stderr, "SMH subsection\n");
        mh::SuperMinHash<policy::SizePow2Policy> smh1(1024), smh2(1024);
        auto start = gett();
        for(unsigned i(0); i < nelem; smh1.addh(i), smh2.addh(i + (nelem/2)), ++i);
        auto end = gett();
        std::fprintf(stderr, "%u insertions (x2) takes %lf us\n", nelem, std::chrono::nanoseconds(end - start).count() / 1000.);
        using f_t = typename mh::SuperMinHash<policy::SizePow2Policy>::final_type;
        f_t fin = smh1.finalize();
        fin.write("tmp.dbb");
        f_t fin2("tmp.dbb");
        f_t fino(smh2.finalize());
        smh1.write("tmp.dbb");
        f_t fin1("tmp.dbb");
        if(std::system("rm tmp.dbb")) throw "up";
        if(std::system("rm SaveSketch.hll")) throw "down";
        std::fprintf(stderr, "ji: %lf with %zu samples packed into %zu words and %zu bits per minimizer\n", fin1.jaccard_index(fino), smh1.size(), fin1.core_.size(), size_t(fin2.b_));
        assert(fin.jaccard_index(fin2) == 1.);
        assert(fin2.jaccard_index(fino) == fin.jaccard_index(fino));
        assert(fin == fin2);
        assert(fin == fin1);
        assert(std::abs(fin1.jaccard_index(fino) - 0.33333333333) <= 0.02);
    }
    {
        std::vector<hll::hll_t> hlls; for(size_t i = 0; i < 10; ++i) hlls.emplace_back(10);
        wy::WyHash<> gen(1337);
        for(size_t i = 0; i < 100000; ++i) {
            if(i % 10 == 0) {
                for(auto &h: hlls) h.addh(gen());
            } else {
                auto v = gen();
                hlls[0].addh(v); hlls[i % 10].addh(v);
            }
        }
        std::vector<hll::hll_t> ohlls; for(size_t i = 0; i < 10; ++i) ohlls.emplace_back(10);
        gzFile fp = gzopen("10hlls.whooo", "wb");
        if(!fp) throw "a party";
        for(auto &h: hlls) h.write(fp);
        gzclose(fp);
        fp = gzopen("10hlls.whooo", "rb");
        for(auto &h: ohlls) h.read(fp);
        gzclose(fp);
        if(std::system("rm 10hlls.whooo")) throw "sideways";
        assert(std::equal(hlls.begin(), hlls.end(), ohlls.begin()));
    }
    {
        sketch::setsketch::EShortSetS ss(100);
        for(size_t i = 0; i < 1000; ++i) ss.add(i);
        ss.write("ss100.ss");
         sketch::setsketch::EShortSetS ss2("ss100.ss");
        assert(ss2 == ss);
        if(std::system("rm ss100.ss")) throw "sideways";
    }
}
