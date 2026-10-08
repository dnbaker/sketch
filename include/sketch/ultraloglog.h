#pragma once
#ifndef SKETCH_ULTRALOGLOG_H__
#define SKETCH_ULTRALOGLOG_H__
/*
 * UltraLogLog: register-compressed successor to HyperLogLog.
 *
 * Implements O. Ertl, "UltraLogLog: A Practical and More Space-Efficient
 * Alternative to HyperLogLog for Approximate Distinct Counting"
 * (arXiv:2308.16862, VLDB 2024), following the production-ready Hash4j
 * reference implementation (github.com/dynatrace-oss/hash4j), with register
 * values redefined by the mapping r -> r + 4p - 8 so that every register is
 * a single byte regardless of the precision parameter p.
 *
 * Properties (as in the paper):
 *  - 8-bit registers holding a 6-bit maximum update value plus 2 extra bits
 *    indicating the two next-smaller update values: 28% lower memory-
 *    variance product than 6-bit HLL under ML estimation.
 *  - commutative, idempotent, mergeable, reducible, constant-time insert.
 *  - estimation: bias-reduced maximum likelihood (default-quality) or the
 *    faster optimal-FGRA estimator; martingale (HIP) estimation is supported
 *    via add(hash, MartingaleEstimator&) for non-distributed streams.
 */
#include "common.h"
#include "hash.h"
#include "mle.h"

namespace sketch {
inline namespace ull {

enum UllEstimationMethod: uint8_t {
    ULL_MLE  = 0,
    ULL_FGRA = 1
};

namespace detail {

// Decodes a register byte into the bit map of observed update values.
// Shift counts follow Java semantics (masked mod 64) as in the reference.
inline uint64_t ull_unpack(uint8_t reg) {
    const uint64_t v = uint64_t(4 | (reg & 3));
    const int sh = int(reg >> 2) - 2; // in [-2, 61]
    return v << (sh & 63);
}

// Encodes a (nonzero) bit map of observed update values into a register byte.
inline uint8_t ull_pack(uint64_t hash_prefix) {
    const int nlz = clz(hash_prefix) + 1; // hash_prefix != 0 by construction
    return uint8_t(((0u - uint32_t(nlz)) << 2) | uint32_t((hash_prefix << nlz) >> 62));
}

// Register change probability scaled by 2^64 (reference: Hash4j
// UltraLogLog.getScaledRegisterChangeProbability).
inline uint64_t ull_scaled_register_change_probability(uint8_t reg, uint32_t p) {
    if(reg == 0) return uint64_t(1) << (64 - p);
    const int k = 1 - int(p) + (reg >> 2);
    const uint64_t m = uint64_t(((reg & 2) | ((reg & 1) << 2)) ^ 7);
    return (m << ((~k) & 63)) >> p;
}

} // namespace detail

template<typename HashStruct=WangHash>
class ultraloglogbase_t {
    // m = 2^p registers, one byte each.
    std::vector<uint8_t, common::Allocator<uint8_t>> core_;
    uint32_t             np_;
    UllEstimationMethod estim_;
    HashStruct             hf_;
    static constexpr uint32_t MAGIC = 0x554c4c31; // "ULL1"
public:
    using final_type = ultraloglogbase_t<HashStruct>;
    using HashType = HashStruct;

    static constexpr uint32_t min_p() {return 3;}
    static constexpr uint32_t max_p() {return 26;}

    template<typename... Args>
    explicit ultraloglogbase_t(uint32_t np, UllEstimationMethod estim, Args &&... args):
        np_(np), estim_(estim), hf_(std::forward<Args>(args)...) {
        if(np < min_p() || np > max_p())
            throw std::invalid_argument(std::string("[ultraloglog] precision parameter p must be in [") +
                                        std::to_string(min_p()) + ", " + std::to_string(max_p()) +
                                        "], got " + std::to_string(np));
        core_.resize(size_t(1) << np);
    }
    explicit ultraloglogbase_t(uint32_t np): ultraloglogbase_t(np, ULL_FGRA) {}
    ultraloglogbase_t() = delete;
    // Non-template; see exaloglogbase_t for why (literal 0 would otherwise
    // bind to the gzFile overload via null-pointer conversion).
    ultraloglogbase_t(const std::string &path): ultraloglogbase_t(min_p(), ULL_FGRA) {
        read(path);
    }
    ultraloglogbase_t(gzFile fp): ultraloglogbase_t(min_p(), ULL_FGRA) {
        read(fp);
    }

    uint32_t p() const {return np_;}
    uint64_t m() const {return uint64_t(1) << np_;}
    size_t size() const {return core_.size();}
    const auto &core() const {return core_;}
    auto &mutable_core() {return core_;}
    const uint8_t *data() const {return core_.data();}
    UllEstimationMethod get_estim() const {return estim_;}
    void set_estim(UllEstimationMethod estim) {estim_ = estim;}
    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(*this), core_.size() * sizeof(core_[0]));
    }
    void clear() noexcept {
        std::memset(core_.data(), 0, core_.size());
    }
    bool is_empty() const {
        for(const uint8_t v: core_) if(v != 0) return false;
        return true;
    }
    bool operator==(const ultraloglogbase_t &o) const {
        return np_ == o.np_ && core_ == o.core_;
    }
    bool operator!=(const ultraloglogbase_t &o) const {return !(*this == o);}

    // Theoretical relative standard errors (asymptotic).
    static double mle_relative_error(uint64_t m) {return 0.7608621002725182 / std::sqrt(double(m));}
    static double fgra_relative_error(uint64_t m) {return std::sqrt(0.6118931496978437 / double(m));}
    double relative_error() const {
        return estim_ == ULL_MLE ? mle_relative_error(m()) : fgra_relative_error(m());
    }

    INLINE uint64_t hash(uint64_t val) const {return hf_(val);}
    INLINE void addh(uint64_t element) noexcept {
        add(hf_(element));
    }
    INLINE void addh(uint64_t element, mle::detail::MartingaleEstimator &martingale) noexcept {
        add(hf_(element), martingale);
    }
    void add(const uint64_t *vals, size_t n) noexcept {
        for(size_t i = 0; i < n; ++i) add(vals[i]);
    }
    template<typename It>
    void add(It b, It e) noexcept {add(&*b, std::distance(b, e));}

    // Updates the sketch with a 64-bit hash value (constant time).
    INLINE void add(uint64_t hashval) noexcept {
        const uint32_t p = np_;
        const uint32_t idx = uint32_t(hashval >> (64 - p));
        const int nlz = clz(~((~hashval) << p)); // in {0, ..., 64-p}
        const uint64_t hash_prefix = detail::ull_unpack(core_[idx]) | (uint64_t(1) << (nlz + p - 1));
        core_[idx] = detail::ull_pack(hash_prefix);
    }

    // Updates the sketch and feeds the state-change probability decrement to
    // a martingale estimator. This must be used for *every* insertion for
    // the martingale estimate to be valid.
    INLINE void add(uint64_t hashval, mle::detail::MartingaleEstimator &martingale) noexcept {
        const uint32_t p = np_;
        const uint32_t idx = uint32_t(hashval >> (64 - p));
        const int nlz = clz(~((~hashval) << p));
        const uint8_t old_state = core_[idx];
        const uint8_t new_state = detail::ull_pack(detail::ull_unpack(old_state) | (uint64_t(1) << (nlz + p - 1)));
        core_[idx] = new_state;
        if(new_state != old_state) {
            martingale.state_changed(mle::detail::u64_as_double(
                    detail::ull_scaled_register_change_probability(old_state, p) -
                    detail::ull_scaled_register_change_probability(new_state, p)) * 0x1p-64);
        }
    }

    // Merge (union). other's precision must be at least this sketch's
    // precision; higher-precision sketches are first reduced (downsized),
    // which is lossless in exactly the same way as recording directly at
    // the lower precision (reducibility property).
    ultraloglogbase_t &add(const ultraloglogbase_t &other) {
        if(other.core_.size() < core_.size())
            throw std::invalid_argument("[ultraloglog] cannot merge a sketch with smaller precision");
        if(other.core_.size() == core_.size()) {
            for(size_t i = 0; i < core_.size(); ++i) {
                if(other.core_[i])
                    core_[i] = detail::ull_pack(detail::ull_unpack(core_[i]) | detail::ull_unpack(other.core_[i]));
            }
        } else {
            const uint32_t p = np_, other_p = other.np_;
            const int other_p_minus_one = int(other_p) - 1;
            const uint64_t k_upper_bound = uint64_t(1) << (other_p - p);
            size_t j = 0;
            for(size_t i = 0; i < core_.size(); ++i) {
                uint64_t hash_prefix = detail::ull_unpack(core_[i]) | detail::ull_unpack(other.core_[j]);
                ++j;
                for(uint64_t k = 1; k < k_upper_bound; ++k) {
                    if(other.core_[j])
                        hash_prefix |= uint64_t(1) << (clz(k) + other_p_minus_one);
                    ++j;
                }
                if(hash_prefix) core_[i] = detail::ull_pack(hash_prefix);
            }
        }
        return *this;
    }
    ultraloglogbase_t &operator+=(const ultraloglogbase_t &other) {return add(other);}
    ultraloglogbase_t operator+(const ultraloglogbase_t &other) const {
        ultraloglogbase_t ret(*this);
        ret.add(other);
        return ret;
    }

    // Returns a copy of this sketch reduced to precision new_p <= p.
    ultraloglogbase_t downsize(uint32_t new_p) const {
        if(new_p < min_p() || new_p > max_p())
            throw std::invalid_argument(std::string("[ultraloglog] invalid precision parameter ") + std::to_string(new_p));
        if(new_p >= np_) return *this;
        return ultraloglogbase_t(new_p, estim_).add(*this);
    }

    // Probability that inserting a new distinct element changes the state.
    double state_change_probability() const {
        uint64_t sum = 0;
        for(const uint8_t v: core_) sum += detail::ull_scaled_register_change_probability(v, np_);
        if(sum == 0 && core_[0] == 0) return 1.; // empty state; sum overflowed to 0
        return mle::detail::u64_as_double(sum) * 0x1p-64;
    }

    double estimate() const {return estim_ == ULL_MLE ? mle_estimate() : fgra_estimate();}
    double report() const {return estimate();}
    double creport() const {return estimate();}
    double cardinality_estimate() const {return estimate();}

    // Bias-reduced maximum likelihood estimate (MVP = 4.63 with 8-bit registers).
    double mle_estimate() const {
        static constexpr double INV_SQRT_FISHER_INFORMATION = 0.7608621002725182;
        static constexpr double ML_EQUATION_SOLVER_EPS = 0.001 * INV_SQRT_FISHER_INFORMATION;
        static constexpr double ML_BIAS_CORRECTION_CONSTANT = 0.48147376527720065;
        uint64_t sum = 0;
        uint32_t b[64]{0};
        for(const uint8_t reg: core_) {
            const int r = reg;
            const int r2 = r - (int(np_) << 2) - 4;
            if(r2 < 0) {
                uint64_t ret = 4;
                if(r2 == -2 || r2 == -8) {b[0] += 1; ret -= 2;}
                if(r2 == -2 || r2 == -4) {b[1] += 1; ret -= 1;}
                sum += ret << (62 - np_);
            } else {
                const int k = r2 >> 2;
                uint64_t ret = 0xE000000000000000ULL;
                const int y0 = r & 1, y1 = (r >> 1) & 1;
                ret -= uint64_t(y0) << 63;
                ret -= uint64_t(y1) << 62;
                b[k] += y0;
                b[k + 1] += y1;
                b[k + 2] += 1;
                sum += ret >> (k + np_);
            }
        }
        const double factor = double(m()) * 2.;
        if(sum == 0) return core_[0] == 0 ? 0.: std::numeric_limits<double>::infinity();
        b[63 - np_] += b[64 - np_];
        const double a = mle::detail::u64_as_double(sum) * factor * 0x1p-64;
        return factor * mle::detail::solve_maximum_likelihood_equation(a, b, 63 - np_, ML_EQUATION_SOLVER_EPS / std::sqrt(double(m())))
               / (1. + ML_BIAS_CORRECTION_CONSTANT / double(m()));
    }

    // Optimal further-generalized-remaining-area (FGRA) estimate: faster,
    // marginally larger error (MVP = 4.94). This is the default estimator.
    double fgra_estimate() const {
        // Constants from the reference implementation (optimal tau).
        static constexpr double ETA_0 = 4.663135422063788;
        static constexpr double ETA_1 = 2.1378502137958524;
        static constexpr double ETA_2 = 2.781144650979996;
        static constexpr double ETA_3 = 0.9824082545153715;
        static constexpr double TAU = 0.8194911375910897;
        static constexpr double MINUS_INV_TAU = -1. / TAU;
        static constexpr double ETA_X = ETA_0 - ETA_1 - ETA_2 + ETA_3;
        static constexpr double ETA23X = (ETA_2 - ETA_3) / ETA_X;
        static constexpr double ETA13X = (ETA_1 - ETA_3) / ETA_X;
        static constexpr double ETA3012XX = (ETA_3 * ETA_0 - ETA_1 * ETA_2) / (ETA_X * ETA_X);
        static constexpr double POW_2_TAU = 1.7641729787046794;  // 2^TAU
        static constexpr double POW_2_MINUS_TAU = 0.5667904887703185; // 2^-TAU
        static constexpr double POW_4_MINUS_TAU = 0.32125077318876137; // 4^-TAU
        static constexpr double POW_4_MINUS_TAU_ETA_23 = POW_4_MINUS_TAU * (ETA_2 - ETA_3);
        static constexpr double POW_4_MINUS_TAU_ETA_01 = POW_4_MINUS_TAU * (ETA_0 - ETA_1);
        static constexpr double POW_4_MINUS_TAU_ETA_3 = POW_4_MINUS_TAU * ETA_3;
        static constexpr double POW_4_MINUS_TAU_ETA_1 = POW_4_MINUS_TAU * ETA_1;
        static constexpr double POW_2_MINUS_TAU_ETA_X = POW_2_MINUS_TAU * ETA_X;
        static constexpr double PHI_1 = ETA_0 / (POW_2_TAU * (2. * POW_2_TAU - 1));
        static constexpr double P_INITIAL = ETA_X * (POW_4_MINUS_TAU / (2. - POW_2_MINUS_TAU));
        static constexpr double POW_2_MINUS_TAU_ETA_02 = POW_2_MINUS_TAU * (ETA_0 - ETA_2);
        static constexpr double POW_2_MINUS_TAU_ETA_13 = POW_2_MINUS_TAU * (ETA_1 - ETA_3);
        static constexpr double POW_2_MINUS_TAU_ETA_2 = POW_2_MINUS_TAU * ETA_2;
        static constexpr double POW_2_MINUS_TAU_ETA_3 = POW_2_MINUS_TAU * ETA_3;
        static const double ESTIMATION_FACTORS[] = {
            94.59941722950778, 455.6358404615186, 2159.476860400962, 10149.51036338182,
            47499.52712820488, 221818.76564766388, 1034754.6840013304, 4824374.384717942,
            2.2486750611989766E7, 1.0479810199493326E8, 4.8837185623048025E8, 2.275794725435168E9,
            1.0604938814719946E10, 4.9417362104242645E10, 2.30276227770117E11, 1.0730444972228585E12,
            5.0001829613164E12, 2.329988778511272E13, 1.0857295240912981E14, 5.059288069986326E14,
            2.3575295235667005E15, 1.0985627213141412E16, 5.119087674515589E16, 2.3853948339571715E17
        };
        // g(r) register contributions, indexed by r - 4p - 4.
        static const double REGISTER_CONTRIBUTIONS[] = {
        0.8484061093359406, 0.38895829052007685, 0.5059986252327467, 0.17873835725405993,
        0.48074234060273024, 0.22040001471443574, 0.2867199572932749, 0.10128061935935387,
        0.2724086914332655, 0.12488785473931466, 0.16246750447680292, 0.057389829555353204,
        0.15435814343988866, 0.0707666752272979, 0.09206087452057209, 0.03251947467566813,
        0.08746577181824695, 0.0400993542020493, 0.05216553700867983, 0.018426892732996067,
        0.04956175987398336, 0.022721969094305374, 0.029559172293066274, 0.01044144713836362,
        0.02808376340530896, 0.012875216815740723, 0.01674946174724118, 0.005916560101748389,
        0.015913433441643893, 0.0072956356627506685, 0.009490944673308844, 0.0033525700962450116,
        0.009017216113341773, 0.004134011914931561, 0.0053779657012946284, 0.0018997062578498703,
        0.005109531310944485, 0.002342503834183061, 0.00304738001114257, 0.001076452918957914,
        0.0028952738727082267, 0.0013273605219527246, 0.0017267728074345586, 6.09963188753462E-4,
        0.0016405831157217021, 7.521379173550258E-4, 9.78461602292084E-4, 3.4563062172237723E-4,
        9.2962292270938E-4, 4.2619276177576713E-4, 5.544372155028133E-4, 1.958487477192352E-4,
        5.267631795945699E-4, 2.4149862146135835E-4, 3.141672858847145E-4, 1.1097608132071735E-4,
        2.9848602115777116E-4, 1.3684320663902123E-4, 1.7802030736817869E-4, 6.288368329501905E-5,
        1.6913464774658265E-4, 7.754107700464113E-5, 1.0087374230011362E-4, 3.563252169014952E-5,
        9.583875639268212E-5, 4.393801322487549E-5, 5.715927601779108E-5, 2.0190875207520577E-5,
        5.430624268457414E-5, 2.4897113642537945E-5, 3.2388833410757184E-5, 1.144099329232623E-5,
        3.0772185549154786E-5, 1.4107744575453657E-5, 1.8352865935237916E-5, 6.482944704957522E-6,
        1.7436805727319977E-5, 7.99403737572986E-6, 1.0399500462555932E-5, 3.67350727106242E-6,
        9.880422483694849E-6, 4.529755498675165E-6, 5.892791363067244E-6, 2.081562667074589E-6,
        5.5986600976661345E-6, 2.5667486794686803E-6, 3.339101736056405E-6, 1.1795003568090263E-6,
        3.1724346748254955E-6, 1.4544270182973653E-6, 1.8920745223756656E-6, 6.683541714686068E-7,
        1.7976340035771381E-6, 8.241391019206623E-7, 1.072128458850476E-6, 3.7871739159788393E-7,
        1.0186145159929963E-6, 4.6699164053601817E-7, 6.075127690181302E-7, 2.1459709360913574E-7,
        5.77189533646426E-7, 2.6461697039041317E-7, 3.442421115430427E-7, 1.2159967724530947E-7,
        3.27059699739513E-7, 1.4994302882644454E-7, 1.9506195985170504E-7, 6.890345650764188E-8,
        1.853256875916027E-7, 8.49639834530526E-8, 1.1053025444979778E-7, 3.904357664636507E-8,
        1.0501327589016596E-7, 4.814414208323267E-8, 6.263105916717392E-8, 2.2123721430020238E-8,
        5.9504908663745294E-8, 2.7280481949286693E-8, 3.548937430686624E-8, 1.2536224699555158E-8,
        3.371796684815404E-8, 1.545826061452554E-8, 2.0109761920695445E-8, 7.103548569567803E-9,
        1.910600846054063E-8, 8.759296176321385E-9, 1.139503111580109E-8, 4.0251673442004705E-9,
        1.082626247715867E-8, 4.963383100969499E-9, 6.456900615837058E-9, 2.28082795382416E-9,
        6.134612546958812E-9, 2.812460192131048E-9, 3.65874960227048E-9, 1.292412391857717E-9,
        3.476127720042246E-9, 1.5936574250689536E-9, 2.0732003554895977E-9, 7.323348470132607E-10,
        1.9697191686598677E-9, 9.030328662369446E-10, 1.1747619217600795E-9, 4.1497151491950363E-10,
        1.1161251587553774E-9, 5.116961428952198E-10, 6.656691762391315E-10, 2.351401942661752E-10,
        6.324431369849931E-10, 2.899484087937328E-10, 3.771959611450379E-10, 1.3324025619025952E-10,
        3.5836869940773545E-10, 1.6429687995368037E-10, 2.1373498756237659E-10, 7.549949478033437E-11,
        2.0306667462222755E-10, 9.309747508122088E-11, 1.2111117194789844E-10, 4.2781167456975155E-11,
        1.1506606020637118E-10, 5.275291818652914E-11, 6.86266490006118E-11, 2.424159650745726E-11,
        6.520123617549523E-11, 2.9892007004129765E-11, 3.888672595026375E-11, 1.3736301184893309E-11,
        3.6945743959497274E-11, 1.693805979747882E-11, 2.2034843273746723E-11, 7.783562034953282E-12,
        2.093500180037604E-11, 9.597812206565218E-12, 1.248586262365167E-11, 4.4104913787558985E-12,
        1.186264650299681E-11, 5.4385213096368525E-12, 7.075011313669894E-12, 2.499168647301308E-12,
        6.721871027139603E-12, 3.081693348317683E-12, 4.008996942969544E-12, 1.4161333491633975E-12,
        3.808892905481426E-12, 1.7462161775917615E-12, 2.271665129027518E-12, 8.024403094117999E-13,
        2.1582778227746425E-12, 9.89479027998621E-13, 1.2872203525845489E-12, 4.54696198313039E-13,
        1.2229703685228866E-12, 5.606801491206791E-13, 7.293928206826874E-13, 2.5764985922987735E-13,
        6.92986095905959E-13, 3.1770479284824887E-13, 4.1330443990824427E-13, 1.4599517261737423E-13,
        3.926748688923721E-13, 1.8002480658009348E-13, 2.3419555992885186E-13, 8.272696321778206E-14,
        2.225059832666067E-13, 1.0200957528418621E-13, 1.327049869160979E-13, 4.687655297461429E-14,
        1.2608118449008524E-13, 5.780288643182276E-14, 7.519618885068399E-14, 2.656221301145837E-14,
        7.144286571105751E-14, 3.2753529955811655E-14, 4.2609301647742677E-14, 1.5051259431302017E-14,
        4.0482511975524363E-14, 1.8559518231526075E-14, 2.4144210160882415E-14, 8.528672304925501E-15,
        2.293908229376684E-14, 1.0516598285774437E-14, 1.3681118012966618E-14, 4.832701981970378E-15,
        1.2998242223663023E-14, 5.959143881034847E-15, 7.752292944665042E-15, 2.7384108113817744E-15,
        7.365346997814574E-15, 3.376699844369893E-15, 4.392773006047039E-15, 1.5516979527951759E-15,
        4.173513269314059E-15, 1.9133791810691354E-15, 2.4891286772044455E-15, 8.792568765435867E-16
        };
        // psi as defined in the paper, divided by ETA_X.
        const auto psi_prime = [](double z, double z_square) {
            return (z + ETA23X) * (z_square + ETA13X) + ETA3012XX;
        };
        const auto sigma = [&](double z) {
            if(z <= 0.) return ETA_3;
            if(z >= 1.) return std::numeric_limits<double>::infinity();
            double pow_z = z;
            double next_pow_z = pow_z * pow_z;
            double s = 0.;
            double pow_tau = ETA_X;
            while(true) {
                const double old_s = s;
                const double next_next_pow_z = next_pow_z * next_pow_z;
                s += pow_tau * (pow_z - next_pow_z) * psi_prime(next_pow_z, next_next_pow_z);
                if(!(s > old_s)) return s / z;
                pow_z = next_pow_z;
                next_pow_z = next_next_pow_z;
                pow_tau *= POW_2_TAU;
            }
        };
        const auto phi = [&](double z, double z_square) {
            if(z <= 0.) return 0.;
            if(z >= 1.) return PHI_1;
            double previous_pow_z = z_square;
            double pow_z = z;
            double next_pow_z = std::sqrt(pow_z);
            double p = P_INITIAL / (1. + next_pow_z);
            double ps = psi_prime(pow_z, previous_pow_z);
            double s = next_pow_z * (ps + ps) * p;
            while(true) {
                previous_pow_z = pow_z;
                pow_z = next_pow_z;
                const double old_s = s;
                next_pow_z = std::sqrt(pow_z);
                const double next_ps = psi_prime(pow_z, previous_pow_z);
                p *= POW_2_MINUS_TAU / (1. + next_pow_z);
                s += next_pow_z * ((next_ps + next_ps) - (pow_z + next_pow_z) * ps) * p;
                if(!(s > old_s)) return s;
                ps = next_ps;
            }
        };
        const auto small_range_estimate = [](uint64_t c0, uint64_t c4, uint64_t c8, uint64_t c10, uint64_t m) {
            const uint64_t alpha = m + 3 * (c0 + c4 + c8 + c10);
            const uint64_t beta = m - c0 - c4;
            const uint64_t gamma = 4 * c0 + 2 * c4 + 3 * c8 + c10;
            const double quad_root_z = (std::sqrt(double(beta * beta + 4 * alpha * gamma)) - double(beta)) / (2. * double(alpha));
            const double root_z = quad_root_z * quad_root_z;
            return root_z * root_z;
        };
        const auto large_range_estimate = [](uint64_t c4w0, uint64_t c4w1, uint64_t c4w2, uint64_t c4w3, uint64_t m) {
            const uint64_t alpha = m + 3 * (c4w0 + c4w1 + c4w2 + c4w3);
            const uint64_t beta = c4w0 + c4w1 + 2 * (c4w2 + c4w3);
            const uint64_t gamma = m + 2 * c4w0 + c4w2 - c4w3;
            return std::sqrt((std::sqrt(double(beta * beta + 4 * alpha * gamma)) - double(beta)) / (2. * double(alpha)));
        };

        uint64_t c0 = 0, c4 = 0, c8 = 0, c10 = 0, c4w0 = 0, c4w1 = 0, c4w2 = 0, c4w3 = 0;
        double sum = 0.;
        const int off = (int(np_) << 2) + 4;
        for(const uint8_t reg: core_) {
            const int r = reg;
            const int r2 = r - off;
            if(r2 < 0) {
                if(r2 < -8) ++c0;
                if(r2 == -8) ++c4;
                if(r2 == -4) ++c8;
                if(r2 == -2) ++c10;
            } else if(r < 252) {
                sum += REGISTER_CONTRIBUTIONS[r2];
            } else {
                if(r == 252) ++c4w0;
                if(r == 253) ++c4w1;
                if(r == 254) ++c4w2;
                if(r == 255) ++c4w3;
            }
        }
        if(c0 || c4 || c8 || c10) {
            const double z = small_range_estimate(c0, c4, c8, c10, m());
            if(c0)  sum += double(c0) * sigma(z);
            if(c4)  sum += double(c4) * POW_2_MINUS_TAU_ETA_X * psi_prime(z, z * z);
            if(c8)  sum += double(c8) * (z * POW_4_MINUS_TAU_ETA_01 + POW_4_MINUS_TAU_ETA_1);
            if(c10) sum += double(c10) * (z * POW_4_MINUS_TAU_ETA_23 + POW_4_MINUS_TAU_ETA_3);
        }
        if(c4w0 || c4w1 || c4w2 || c4w3) {
            const int w = 65 - int(np_);
            const double z = large_range_estimate(c4w0, c4w1, c4w2, c4w3, m());
            const double root_z = std::sqrt(z);
            double s = phi(root_z, z) * double(c4w0 + c4w1 + c4w2 + c4w3);
            s += z * (1. + root_z) * (double(c4w0) * ETA_0 + double(c4w1) * ETA_1 + double(c4w2) * ETA_2 + double(c4w3) * ETA_3);
            s += root_z * ((double(c4w0 + c4w1) * (z * POW_2_MINUS_TAU_ETA_02 + POW_2_MINUS_TAU_ETA_2))
                         + (double(c4w2 + c4w3) * (z * POW_2_MINUS_TAU_ETA_13 + POW_2_MINUS_TAU_ETA_3)));
            sum += s * std::pow(POW_2_MINUS_TAU, w) / ((1. + root_z) * (1. + z));
        }
        return ESTIMATION_FACTORS[np_ - min_p()] * std::pow(sum, MINUS_INV_TAU);
    }

    std::string desc_string() const {
        return std::string("UltraLogLog, p: ") + std::to_string(np_) + ", m: " + std::to_string(m())
             + ", relerr: " + std::to_string(relative_error());
    }
    std::string to_string() const {
        return desc_string() + ", estimate: " + std::to_string(estimate());
    }

    void write(gzFile fp) const {
#define CW(fp, src, len) do {if(gzwrite(fp, src, len) == 0) throw std::runtime_error("Error writing to file.");} while(0)
        uint32_t header[4]{MAGIC, np_, estim_, 0};
        CW(fp, header, sizeof(header));
        CW(fp, core_.data(), core_.size() * sizeof(core_[0]));
#undef CW
    }
    void write(const char *path) const {
        gzFile fp(gzopen(path, "wb"));
        if(!fp) throw ZlibError(Z_ERRNO, std::string("Could not open file at '") + path + "' for writing");
        write(fp);
        gzclose(fp);
    }
    void write(const std::string &path) const {write(path.data());}
    void read(gzFile fp) {
#define CR(fp, dst, len) \
    do {\
        if(static_cast<uint64_t>(gzread(fp, dst, len)) != len) {\
            throw ZlibError(std::string("[E:") + __FILE__ + ':' + std::to_string(__LINE__) + ':' + __PRETTY_FUNCTION__ + "] Error reading from file");\
        }\
    } while(0)
        uint32_t header[4];
        CR(fp, header, sizeof(header));
        if(header[0] != MAGIC) throw std::runtime_error("[ultraloglog] bad magic: not an UltraLogLog sketch");
        if(header[1] < min_p() || header[1] > max_p())
            throw std::runtime_error(std::string("[ultraloglog] invalid precision parameter ") + std::to_string(header[1]));
        np_ = header[1];
        estim_ = UllEstimationMethod(header[2]);
        core_.resize(size_t(1) << np_);
        CR(fp, core_.data(), core_.size() * sizeof(core_[0]));
#undef CR
    }
    void read(const char *path) {
        gzFile fp(gzopen(path, "rb"));
        if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at '") + path + "' for reading");
        read(fp);
        gzclose(fp);
    }
    void read(const std::string &path) {read(path.data());}
};
using ultraloglog_t = ultraloglogbase_t<>;

} // namespace ull
} // namespace sketch

#endif // SKETCH_ULTRALOGLOG_H__
