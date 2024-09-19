#include <iostream>
#include <iomanip>
#include "Variable.hpp"
#include "Dual.hpp"


template<typename T>
T f(const T& x, const T& y) {
    auto tmp = (x.log() + (-x) * y - y.sin());
    auto tmp2 = tmp;
    for (int i = 1; i < 5; ++i)
        tmp = tmp * static_cast<T>(i);
    return tmp / ((static_cast<T>(2) * x).cos() + tmp2).exp();
}

int main(int argc, char const *argv[])
{
    std::cout << std::setprecision(10);

    using dtype = _Float32;

    Dual<dtype> x1(2, 1), x2(5, 0);
    auto out_dual = f(x1, x2);
    std::cout << out_dual << std::endl;

    x1 = Dual<dtype>(2, 0);
    x2 = Dual<dtype>(5, 1);
    out_dual = f(x1, x2);
    std::cout << out_dual << std::endl;

    Variable<dtype> x(2, true), y(5, true);
    auto out = f(x, y);
    out.backward(1, false);
    std::cout << out << std::endl;
    std::cout << x << std::endl;
    std::cout << y << std::endl;



    // Dual<dtype> x1(2, 1), x2(5, 0);
    // auto fwd_out = x1.log() + x1 * x2 - x2.sin();
    // std::cout << fwd_out << std::endl;

    // x1 = Dual<dtype>(2, 0);
    // x2 = Dual<dtype>(5, 1);
    // fwd_out = x1.log() + x1 * x2 - x2.sin();
    // std::cout << fwd_out << std::endl;

    // Variable<dtype> v1(2, true), v2(5, true);
    // auto bwd_out = v1.log() + v1 * v2 - v2.sin();
    // // auto bwd_out = (v1.log() + (-v1) * v2 - v2.sin()) / ((2.f * v1).cos() + 1.f).exp();
    // bwd_out.backward(1, false);
    // std::cout << bwd_out << std::endl;
    // std::cout << v1 << std::endl;
    // std::cout << v2 << std::endl;

    return 0;
}
