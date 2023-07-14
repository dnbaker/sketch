#define NO_BLAZE
#include "hll.h"
#include "hash.h"

using namespace sketch::hll;
using namespace sketch::hash;
using namespace sketch;

using Hasher = hash::XORConstantHasher<>;

int main() {
    std::mt19937_64 mt(137);
    static constexpr size_t nelem = 1 << 24;
    uint64_t *p = (uint64_t *)std::malloc(8 * nelem);
    if(!p) throw 1;
    for(uint64_t *p2 = p, *e = p + nelem; p2 != e; *p2++ = mt());
    hllbase_t<Hasher> h(16);
    hllbase_t<hash::FusedReversible3<InvMul, InvRShiftXor<33>, MultiplyAddXoRot<13> > > h2(16, ERTL_MLE, ERTL_JOINT_MLE, 1337);
    for(uint64_t *p2 = p, *e = p + nelem; p2 != e; h2.addh(*p2), h.addh(*p2++));
    std::fprintf(stderr, "Random xor 'hash' function estimate for %zu items: %lf\n", nelem, h.report());
    std::fprintf(stderr, "Random hash::FusedReversible3<InvMul, InvRShiftXor<33>, MultiplyAddXoRot<13>> function estimate for %zu items: %lf\n", nelem, h2.report());
}
