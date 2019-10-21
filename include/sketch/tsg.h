#ifndef THREAD_SEEDED_GEN_H__
#define THREAD_SEEDED_GEN_H__
#include <thread>
#include <utility>

namespace tsg {
template<typename RNG>
struct ThreadSeededGen: public RNG {
    template<typename...Args>
    ThreadSeededGen(Args &&...args): RNG(std::forward<Args>(args)...) {
        this->seed(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }
    template<typename...Args>
    decltype(auto) operator()(Args &&...args) {return RNG::operator()(std::forward<Args>(args)...);}
    template<typename...Args>
    decltype(auto) operator()(Args &&...args) const {return RNG::operator()(std::forward<Args>(args)...);}
};

} // tsg

#endif
