#include "sketch/hll.h"
#include <fstream>

void usage() {
    std::fprintf(stderr, "demo: summarize k-mers\n\ndemo <opts> [1.fq 2.fq ...]\n-k: set k [21]\n-p: set p [14]\n");
}

sketch::hll_t sketch_file(std::string path, int p, int k) {
    sketch::hll_t ret(p);
    std::ifstream ifs(path);
    for(std::string line;std::getline(ifs, line);) {
        std::ptrdiff_t n = std::ptrdiff_t(line.size()) - k + 1;
        for(std::ptrdiff_t i = 0; i < n; ++i) {
            ret.add(XXH3_64bits(line.data() + i, k));
        }
    }
    return ret;
}

int main(int argc, char *argv[]) {
    int k = 21, p = 14;
    for(int c;(c = getopt(argc, argv, "p:k:h?")) >= 0;) {switch(c) {
        case 'k': k = std::atoi(argv[optind]); break;
        case 'p': p = std::atoi(argv[optind]); break;
        default: case 'h': usage(); std::exit(1);
    }}
    std::vector<sketch::hll_t> sketches;
    for(char **a = argv + optind; *a; ++a) {
        sketches.emplace_back(sketch_file(*a, p, k));
        std::fprintf(stderr, "Path %s has %g cardinality\n", *a, sketches.back().report());
    }
}
