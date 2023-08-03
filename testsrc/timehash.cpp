#define DUMMY_INVERSE 1
#include "hash.h"
#include "common.h"
using namespace sketch;
using namespace hash;

// from Facebook's Folly
template <typename T> void doNotOptimizeAway(const T& datum) {asm volatile("" ::"r"(datum));}

#if 1
#define CALL_HASH(hasher, i) hasher(hasher.inverse(i))
#elif 1
#define CALL_HASH(hasher, i) hasher.inverse(i)
#else
#define CALL_HASH(hasher, i) hasher(i)
#endif


int main() {
    WangHash hash;
    uint64_t seed1 = hash(uint64_t(1337));
    hash::MultiplyAddXoRot<33> gen1(seed1, hash(seed1));
    hash::MultiplyAddXor gen2(seed1);
    hash::XorMultiply gen3(seed1, hash(seed1));
    hash::FusedReversible3<InvMul, RotL33, MultiplyAddXoRot<16>> gen4(seed1, hash(seed1));
    XorMultiply gen5(seed1, hash(seed1));
    hash::FusedReversible<RotL33, MultiplyAdd> gen__5(1337);
    hash::KWiseIndependentPolynomialHash<4> fivewise_gamgee;
    hash::FusedReversible3<InvMul, InvRShiftXor<33>, MultiplyAddXoRot<13>> gen8(hash(hash(seed1)), hash(1337));
    hash::FusedReversible<InvRShiftXor<33>, InvMul> gen9(hash(hash(seed1)), hash(1337));
    hash::FusedReversible<InvRShiftXor<33>, MultiplyAddXor> gen10(hash(hash(seed1)), hash(1337));
    MurFinHash mfh;
    hash::KWiseIndependentPolynomialHash61<4> fivewise_nomee;
    std::array<size_t, 11> arr{0};
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t accum = 0;
    uint64_t nelem = 100000000;
    for(uint64_t i = 0; i < nelem; ++i) {
        doNotOptimizeAway(CALL_HASH(hash, i));
    }
    auto end = std::chrono::high_resolution_clock::now();
    arr[0] = size_t(std::chrono::nanoseconds(end - start).count());
    std::fprintf(stderr, "diff: %zu. accum %zu\n", size_t(std::chrono::nanoseconds(end - start).count()), size_t(accum));
#define DO_THING(hasher, namer, ind) \
    start = std::chrono::high_resolution_clock::now();\
    accum = 0;\
    for(uint64_t i = 0; i < nelem; ++i) {\
       doNotOptimizeAway(CALL_HASH(hasher, i));\
    }\
    end = std::chrono::high_resolution_clock::now();\
    arr[ind] = size_t(std::chrono::nanoseconds(end - start).count());\
    std::fprintf(stderr, "for %s diff: %zu. accum %zu\n", namer, size_t(std::chrono::nanoseconds(end - start).count()), size_t(accum));

    DO_THING(gen1, "multiplyaddxorot33", 1)
    DO_THING(gen2, "mulyaddor", 2)
    DO_THING(gen3, "xormult", 3)
    DO_THING(gen4, "rotxorrot", 4)
    DO_THING(gen__5, "xormltiply", 5)
    DO_THING(fivewise_gamgee, "fivewise", 5)
    DO_THING(fivewise_nomee, "fivewise61", 5)
    DO_THING(gen5, "mul-bf-rrot31", 6)
    DO_THING(mfh, "murfinhash", 7)
    DO_THING(gen8, "invmul, invrshift33, maxorot", 8)
    DO_THING(gen9, "invrshiftxor33, invmul", 9)
    DO_THING(gen10, "invrshiftxor33, multiplyaddxor (add,mul,xor)", 10)
    for(size_t i = 0; i < 100000; ++i) {
        assert(gen3.inverse(gen3(i)) == i);
        assert(gen2.inverse(gen2(i)) == i);
        assert(gen1.inverse(gen1(i)) == i);
        assert(gen__5.inverse(gen__5(i)) == i);
        assert(gen8.inverse(gen8(i)) == i);
        assert(gen9.inverse(gen9(i)) == i);
        assert(gen10.inverse(gen10(i)) == i);
    }
    for(size_t i = 1; i < arr.size(); ++i)
        std::fprintf(stderr, "%zu is %lf as fast as WangHash\n", i, double(arr[0]) / arr[i]);
}
