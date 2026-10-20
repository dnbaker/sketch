// Head-to-head at hash-table scale: our cqfbase_t / bcqf_t / cht_t vs
// ankerl::unordered_dense. q=22 (4M slots), 48-bit reduced keys, 16-bit counts;
// fills 0.5/0.9; q sweep at 20/22/24 (fill 0.9).
//
// KEY CONTRACT (all our filters): keys are PRE-HASHED. add()/insert
// take KEY_BITS-wide reduced hashes; bits above KEY_BITS are ignored by
// construction (bcqf masks hv &= HVMASK at every entry; cqfbase_t's
// parts() masks the quotient and remainder; cht_t masks hv &= HVMASK in
// home()/fp()). Miss keys deliberately include ~key (top bits set) to
// exercise that contract.
//
// An external reference quotient-filter implementation exists for
// out-of-tree comparison; per project policy it is not vendored or
// referenced beyond this note.
//
// MEMORY ACCOUNTING (bits/entry = bytes*8 / keys inserted):
//   cqf/bcqf  est_memory_usage().second (payload + metadata + dirs)
//   cht       bytes_used() (packed rem + packed cnt + occupancy bitmap;
//             exact array accounting)
//   umap      analytic basis, ankerl::unordered_dense v4.1 (vendored at
//             include/ankerl): dense m_values holds n x sizeof(pair<u64,u16>)
//             = 16 B/entry (no holes, no per-node malloc), plus
//             bucket_count() x sizeof(bucket_entry) for the open-addressing
//             bucket array (m_value_idx + m_distance = 8 B). Stated so the
//             comparison is auditable.
//
// cht_t (include/sketch/cht.h) is generic over reversible hashes
// (cht_identity_hash<48> here -- the pre-hashed reduced-key contract, a
// drop-in for cqfbase_t call sites; add() applies the hash internally and
// the identity masks, exactly like cqfbase_t's parts()). Its bulk build,
// add_bulk_sorted, consumes keys sorted by hash(key) ascending, i.e. by
// the (home, fingerprint) pair -- for the identity hash that is the sort
// by masked-key bits [KEY_BITS-q-rs, KEY_BITS) this harness feeds every
// filter's bulk path (a plain masked-key sort in exact mode, rs = r).
#include "cqf.h"
#include "cht.h"
#include "ankerl/unordered_dense.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>

using namespace sketch;
using clk = std::chrono::high_resolution_clock;
static double now_s() {
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}
static double bits_per_entry(size_t bytes, uint64_t n) { return double(bytes) * 8. / double(n); }

struct Ops { double ins, hit, miss, inc; uint64_t nhits, fph; };
// insert [0,target) from ins, then qhit/qmiss/inc over the same prefix
template<typename F>
static Ops time_ops(F &f, const uint64_t *ins, const std::vector<uint64_t> &qkeys,
                    const std::vector<uint64_t> &miss, uint64_t target) {
    Ops r;
    double t0 = now_s();
    for (uint64_t i = 0; i < target; ++i) f.add(ins[i]);
    double t1 = now_s();
    uint64_t hit = 0;
    for (uint64_t i = 0; i < target; ++i) hit += f.count(qkeys[i]) > 0;
    double t2 = now_s();
    uint64_t fph = 0;
    for (uint64_t i = 0; i < target; ++i) fph += f.count(miss[i]) > 0;
    double t3 = now_s();
    for (uint64_t i = 0; i < target; ++i) f.add(qkeys[i]);  // inc on existing
    double t4 = now_s();
    r.ins = target / (t1 - t0) / 1e6; r.hit = target / (t2 - t1) / 1e6;
    r.miss = target / (t3 - t2) / 1e6; r.inc = target / (t4 - t3) / 1e6;
    r.nhits = hit; r.fph = fph;
    return r;
}
static void print_ops(const char *tag, const Ops &r, size_t bytes, uint64_t target) {
    std::printf("  %-10s insert %7.3f M/s  qhit %7.3f M/s (%llu)  qmiss %7.3f M/s (fp %llu)  inc %7.3f M/s  [%zu B, %.1f bits/entry]\n",
                tag, r.ins, r.hit, (unsigned long long)r.nhits, r.miss,
                (unsigned long long)r.fph, r.inc, bytes, bits_per_entry(bytes, target));
}

// cht single-insert + sorted-bulk build, exact or truncated (RSTORE_BITS).
template<size_t RS>
static void run_cht(uint32_t q, const std::vector<uint64_t> &keys, uint64_t target,
                    const std::vector<uint64_t> &qkeys, const std::vector<uint64_t> &miss,
                    double fill, const char *tag) {
    const size_t KEY_BITS = 48;
    // (home, fingerprint) order = the masked key's bits [r-rs, KEY_BITS);
    // exact mode (rs == r == KEY_BITS - q) reduces to a masked-key sort
    const unsigned fsh = unsigned(KEY_BITS - q - std::min<size_t>(size_t(RS), KEY_BITS - q));
    std::vector<uint64_t> bkeys(keys.begin(), keys.begin() + target);
    std::sort(bkeys.begin(), bkeys.end(), [fsh](uint64_t a, uint64_t b){ return (a >> fsh) < (b >> fsh); });
    {
        cht_t<uint64_t, cht_identity_hash<48>, 16, RS> f(q);
        std::fprintf(stderr, "[%s single fill %.2f]\n", tag, fill);
        Ops r = time_ops(f, bkeys.data(), qkeys, miss, target);
        print_ops(tag, r, f.bytes_used(), target);
    }
    {
        cht_t<uint64_t, cht_identity_hash<48>, 16, RS> f(q);
        std::fprintf(stderr, "[%s bulk fill %.2f]\n", tag, fill);
        double t0 = now_s();
        f.add_bulk_sorted(bkeys.begin(), bkeys.end());
        double t1 = now_s();
        uint64_t hit = 0;
        for (uint64_t i = 0; i < target; ++i) hit += f.count(qkeys[i]) > 0;
        double t2 = now_s();
        uint64_t fph = 0;
        for (uint64_t i = 0; i < target; ++i) fph += f.count(miss[i]) > 0;
        double t3 = now_s();
        for (uint64_t i = 0; i < target; ++i) f.add(qkeys[i]);  // inc on existing
        double t4 = now_s();
        std::printf("  %-10s bulk-build %6.3f M/s  qhit %7.3f M/s (%llu)  qmiss %7.3f M/s (fp %llu, fpr %.4f)  inc %7.3f M/s  [%zu B, %.1f bits/entry]\n",
                    tag, target / (t1 - t0) / 1e6, target / (t2 - t1) / 1e6,
                    (unsigned long long)hit, target / (t3 - t2) / 1e6,
                    (unsigned long long)fph, double(fph) / double(target),
                    target / (t4 - t3) / 1e6, f.bytes_used(), bits_per_entry(f.bytes_used(), target));
    }
}

int main() {
    setbuf(stderr, 0);
    const uint32_t q = 22;
    const uint64_t nslots = 1ull << q;
    const uint64_t KEY_BITS = 48, VAL_BITS = 16;
    std::mt19937_64 rng(20261016);
    std::vector<uint64_t> keys(nslots);
    for (auto &k : keys) k = rng() & ((1ull << KEY_BITS) - 1);
    std::shuffle(keys.begin(), keys.end(), rng);
    std::vector<uint64_t> qkeys;  // rebuilt per fill: shuffle of the inserted prefix
    std::vector<uint64_t> miss(nslots);
    for (auto &k : miss) k = rng() & ((1ull << KEY_BITS) - 1);
    // half the miss keys forced absent; ~key sets bits >= 48 (wide-hv contract)
    for (size_t i = 0; i < miss.size(); i += 2) miss[i] = ~keys[i];

    for (double fill : {0.5, 0.9}) {
        uint64_t target = uint64_t(fill * nslots);
        qkeys.assign(keys.begin(), keys.begin() + target);
        std::shuffle(qkeys.begin(), qkeys.end(), rng);
        std::printf("=== fill %.2f (%llu keys, q=%u, key%llu/val%llu) ===\n", fill,
                    (unsigned long long)target, q, (unsigned long long)KEY_BITS,
                    (unsigned long long)VAL_BITS);

        // ---- ours: cqf (bulk build in home order) ----
        {
            std::vector<uint64_t> skeys(keys.begin(), keys.begin() + target);
            std::sort(skeys.begin(), skeys.end(), [&](uint64_t a, uint64_t b){return (a >> (KEY_BITS - q)) < (b >> (KEY_BITS - q));});
            cqfbase_t<48,16> f(q);
            std::fprintf(stderr, "[cqf fill %.2f]\n", fill);
            Ops r = time_ops(f, skeys.data(), qkeys, miss, target);
            print_ops("our-cqf", r, f.est_memory_usage().second, target);
        }
        // ---- ours: bcqf (bulk build in home order) ----
        {
            std::vector<uint64_t> skeys(keys.begin(), keys.begin() + target);
            std::sort(skeys.begin(), skeys.end(), [&](uint64_t a, uint64_t b){return (a >> (KEY_BITS - q)) < (b >> (KEY_BITS - q));});
            bcqf48_16 f(q);
            std::fprintf(stderr, "[bcqf fill %.2f]\n", fill);
            Ops r = time_ops(f, skeys.data(), qkeys, miss, target);
            print_ops("our-bcqf", r, f.est_memory_usage().second, target);
        }
        // ---- ours: cqf truncated RSTORE=12 (full ops; the fingerprint
        //      filter mode, cf. cht-t12) ----
        {
            std::vector<uint64_t> skeys(keys.begin(), keys.begin() + target);
            std::sort(skeys.begin(), skeys.end(), [&](uint64_t a, uint64_t b){return (a >> (KEY_BITS - q - 12)) < (b >> (KEY_BITS - q - 12));});
            cqfbase_t<48,16,uint64_t,WangHash,12> f(q);
            std::fprintf(stderr, "[cqf-t12 fill %.2f]\n", fill);
            Ops r = time_ops(f, skeys.data(), qkeys, miss, target);
            print_ops("our-cqf-t12", r, f.est_memory_usage().second, target);
        }
        // ---- ours: cht exact + truncated (single insert and sorted bulk) ----
        run_cht<0>(q, keys, target, qkeys, miss, fill, "our-cht");
        run_cht<12>(q, keys, target, qkeys, miss, fill, "our-cht-t12");
        // ---- ankerl::unordered_dense (memory basis: 16 B value + 8 B bucket) ----
        {
            ankerl::unordered_dense::map<uint64_t, uint16_t> f;
            f.reserve(target * 2);
            double t0 = now_s();
            for (uint64_t i = 0; i < target; ++i) f.emplace(keys[i], 1);
            double t1 = now_s();
            uint64_t hit = 0;
            for (uint64_t i = 0; i < target; ++i) hit += f.contains(qkeys[i]);
            double t2 = now_s();
            uint64_t fph = 0;
            for (uint64_t i = 0; i < target; ++i) fph += f.contains(miss[i]);
            double t3 = now_s();
            for (uint64_t i = 0; i < target; ++i) ++f[qkeys[i]];
            double t4 = now_s();
            const size_t ubytes = f.size() * sizeof(std::pair<uint64_t, uint16_t>)
                                + f.bucket_count() * 8;
            std::printf("  %-10s insert %7.3f M/s  qhit %7.3f M/s (%llu)  qmiss %7.3f M/s (fp %llu)  inc %7.3f M/s  [%zu B, %.1f bits/entry]\n",
                        "umap-dense", target / (t1 - t0) / 1e6, target / (t2 - t1) / 1e6,
                        (unsigned long long)hit, target / (t3 - t2) / 1e6,
                        (unsigned long long)fph, target / (t4 - t3) / 1e6,
                        ubytes, bits_per_entry(ubytes, target));
        }
        // ---- memory-only filter instantiations at this fill ----
        {
            cqfbase_t<48,16,uint64_t,WangHash,8> f8(q);
            cqfbase_t<48,16,uint64_t,WangHash,12> f12(q);
            cht_t<uint64_t, cht_identity_hash<48>, 16, 8> c8(q);
            cht_t<uint64_t, cht_identity_hash<48>, 16, 12> c12(q);
            cht_t<uint64_t, cht_identity_hash<48>, 16, 0, true, 0> cm0(q); // membership-only: no disp array
            // at-fill analytic FPRs: cqf's fpr_bound is the idealized
            // per-(home,fp) value; cht's is the clustering-aware probe-span
            // value (cht probes compare remainders across every slot in the
            // span, not just same-home slots) -- the measured fp columns
            // elsewhere are the apples-to-apples comparison
            const double Lf = (1. + 1. / (1. - fill) / (1. - fill)) / 2.;
            std::printf("  [memory only, fill %.2f] cqf-rs8 %zu B (%.1f b/e, fpr@fill(ideal) %.2e)  cqf-rs12 %zu B (%.1f b/e, fpr@fill(ideal) %.2e)  cht-rs8 %zu B (%.1f b/e, fpr@fill(span) %.2e)  cht-rs12 %zu B (%.1f b/e, fpr@fill(span) %.2e)  cht-noDecode %zu B (%.1f b/e)\n",
                        fill,
                        f8.est_memory_usage().second, bits_per_entry(f8.est_memory_usage().second, target), 1. - std::exp(-double(target) / std::ldexp(1., int(q + 8))),
                        f12.est_memory_usage().second, bits_per_entry(f12.est_memory_usage().second, target), 1. - std::exp(-double(target) / std::ldexp(1., int(q + 12))),
                        c8.bytes_used(), bits_per_entry(c8.bytes_used(), target), 1. - std::exp(-Lf / 256.),
                        c12.bytes_used(), bits_per_entry(c12.bytes_used(), target), 1. - std::exp(-Lf / 4096.),
                        cm0.bytes_used(), bits_per_entry(cm0.bytes_used(), target));
        }
        std::fflush(stdout);
    }
    // ---- q sweep at fill 0.9: bits/entry scaling (and insert M/s) ----
    std::printf("=== q sweep (fill 0.9): slot encoding saves more as the table grows ===\n");
    for (uint32_t qq : {20u, 22u, 24u}) {
        const uint64_t ns = 1ull << qq;
        const uint64_t tgt = uint64_t(0.9 * ns);
        std::vector<uint64_t> k2(tgt);
        for (auto &k : k2) k = rng() & ((1ull << KEY_BITS) - 1);
        std::printf("-- q=%u (%llu slots, %llu keys) --\n", qq, (unsigned long long)ns, (unsigned long long)tgt);
        {   // cht exact: single + bulk
            std::vector<uint64_t> s2(k2);
            std::sort(s2.begin(), s2.end());
            cht_t<uint64_t, cht_identity_hash<48>, 16> f(qq);
            double t0 = now_s();
            for (uint64_t i = 0; i < tgt; ++i) f.add(s2[i]);
            double t1 = now_s();
            std::printf("  %-10s insert %7.3f M/s  [%zu B, %.1f bits/entry]\n", "our-cht",
                        tgt / (t1 - t0) / 1e6, f.bytes_used(), bits_per_entry(f.bytes_used(), tgt));
            cht_t<uint64_t, cht_identity_hash<48>, 16> fb(qq);
            double t2 = now_s();
            fb.add_bulk_sorted(s2.begin(), s2.end());
            double t3 = now_s();
            std::printf("  %-10s bulk-build %6.3f M/s  [%zu B, %.1f bits/entry]\n", "our-cht",
                        tgt / (t3 - t2) / 1e6, fb.bytes_used(), bits_per_entry(fb.bytes_used(), tgt));
        }
        {   // cht truncated rs=12: bulk only (memory story)
            const unsigned fsh = unsigned(KEY_BITS - qq - 12);
            std::vector<uint64_t> s2(k2);
            std::sort(s2.begin(), s2.end(), [fsh](uint64_t a, uint64_t b){ return (a >> fsh) < (b >> fsh); });
            cht_t<uint64_t, cht_identity_hash<48>, 16, 12> f(qq);
            double t0 = now_s();
            f.add_bulk_sorted(s2.begin(), s2.end());
            double t1 = now_s();
            std::printf("  %-10s bulk-build %6.3f M/s  [%zu B, %.1f bits/entry, fpr_bound %.2e]\n", "our-cht-t12",
                        tgt / (t1 - t0) / 1e6, f.bytes_used(), bits_per_entry(f.bytes_used(), tgt), f.fpr_bound());
        }
        {   // cqf exact: construct-only memory (payload is fill-independent)
            cqfbase_t<48,16> f(qq);
            std::printf("  %-10s [construct only]  [%zu B, %.1f bits/entry]\n", "our-cqf",
                        f.est_memory_usage().second, bits_per_entry(f.est_memory_usage().second, tgt));
            cqfbase_t<48,16,uint64_t,WangHash,12> f12(qq);
            std::printf("  %-10s [construct only]  [%zu B, %.1f bits/entry, fpr_bound %.2e]\n", "our-cqf-t12",
                        f12.est_memory_usage().second, bits_per_entry(f12.est_memory_usage().second, tgt), f12.fpr_bound());
        }
        {   // umap-dense basis: reserve only (analytic count, see header)
            ankerl::unordered_dense::map<uint64_t, uint16_t> f;
            f.reserve(tgt * 2);
            const size_t ubytes = tgt * sizeof(std::pair<uint64_t, uint16_t>) + f.bucket_count() * 8;
            std::printf("  %-10s [reserve only]  [%zu B, %.1f bits/entry, %llu buckets]\n", "umap-dense",
                        ubytes, bits_per_entry(ubytes, tgt), (unsigned long long)f.bucket_count());
        }
        std::fflush(stdout);
    }
    // ---- decode path: the reversible-hash dumper (jellyfish property).
    // Same distinct-key stream into both tables; the identity instance
    // consumes it under the pre-hashed reduced-key contract (garbage top
    // bits masked internally), the splitmix64 instance consumes raw 64-bit
    // keys. dump_keys() must return each table's expected key set exactly.
    std::printf("=== decode path (cht dumper), q=%u, fill 0.9 ===\n", q);
    {
        const uint64_t n = uint64_t(0.9 * nslots);
        std::vector<uint64_t> raw(n);
        for (auto &v : raw) v = rng();
        std::sort(raw.begin(), raw.end());
        raw.erase(std::unique(raw.begin(), raw.end()), raw.end());
        const uint64_t nd = raw.size();
        const uint64_t mask48 = (1ull << KEY_BITS) - 1;
        cht_t<uint64_t, cht_identity_hash<48>, 16> fid(q);
        for (uint64_t i = 0; i < nd; ++i) fid.add(raw[i]);
        cht_t<uint64_t, cht_splitmix64_hash, 16> fmix(q);
        for (uint64_t i = 0; i < nd; ++i) fmix.add(raw[i]);
        // expected sets: identity decodes the MASKED stream; splitmix
        // decodes the raw stream exactly (bijection)
        std::vector<uint64_t> exp_id(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) exp_id[i] = raw[i] & mask48;
        std::sort(exp_id.begin(), exp_id.end());
        exp_id.erase(std::unique(exp_id.begin(), exp_id.end()), exp_id.end());
        auto run_dump = [&](const char *tag, auto &f, const std::vector<uint64_t> &expected) {
            std::vector<uint64_t> dumped;
            dumped.reserve(expected.size());
            double t0 = now_s();
            f.dump_keys(std::back_inserter(dumped));
            double t1 = now_s();
            std::sort(dumped.begin(), dumped.end());
            // every dumped key decodes to a key the table reports present,
            // and the counters sum to the insert count
            uint64_t cntsum = 0, present = 0;
            for (auto it = f.begin(); it != f.end(); ++it) {cntsum += it.mapped(); present += f.contains(*it);}
            const bool ok = dumped == expected && cntsum == nd && dumped.size() == f.size() && present == dumped.size();
            std::printf("  %-14s dump %7.3f M keys/s  [%zu keys, set-equality+contains %s, sum(cnt) %llu == inserts %llu]\n",
                        tag, nd / (t1 - t0) / 1e6, dumped.size(),
                        ok ? "OK" : "FAILED", (unsigned long long)cntsum, (unsigned long long)nd);
            return ok;
        };
        const bool ok1 = run_dump("cht-identity", fid, exp_id);
        const bool ok2 = run_dump("cht-splitmix64", fmix, raw);
        std::printf("  decode verification: %s\n", (ok1 && ok2) ? "PASS" : "FAIL");
        std::fflush(stdout);
    }
    return 0;
}
