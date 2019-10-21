#include "policy.h"
#include "exception.h"
#include <cassert>

int main() {
    PREC_REQ(1 == 1, "one must be one");
    for(size_t i = 100; i < 100000; ++i) {
        sketch::policy::SizePow2Policy<uint64_t> p(i);
        sketch::policy::SizeDivPolicy<uint64_t> pd(i);
        assert(p.nelem() == sketch::roundup(i));
        assert(pd.nelem() == i);
    }
}
