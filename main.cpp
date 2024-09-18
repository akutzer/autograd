#include <iostream>
// #include "variable.hpp"
#include "Variable.hpp"
// #include "variable.hpp"
#include "dual.hpp"



int main(int argc, char const *argv[])
{

    Dual<float> x1(2, 1), x2(5, 0);
    auto fwd_out = x1.log() + x1 * x2 - x2.sin();
    std::cout << fwd_out << std::endl;

    Dual<float> xx1(2, 0), xx2(5, 1);
    auto fwd_out2 = xx1.log() + xx1 * xx2 - xx2.sin();
    std::cout << fwd_out2 << std::endl;



    Variable<float> v1(2, true), v2(5, true);

    // auto vv1 = v1.log();
    // auto vv2 = v1 * v2;
    // auto vv3 = v2.sin();
    // auto vv4 = vv1 + vv2;
    // auto vv5 = vv4 - vv3;
    // // auto vv5 = (vv1 + vv2) - vv3;
    
    // vv5.backward();
    // std::cout << vv5 << std::endl;
    // std::cout << vv4 << std::endl;
    // std::cout << vv3 << std::endl;
    // std::cout << vv2 << std::endl;
    // std::cout << vv1 << std::endl;
    // std::cout << v1 << std::endl;
    // std::cout << v2 << std::endl;

    auto bwd_out = v1.log() + v1 * v2 - v2.sin();
    bwd_out.backward(1, false);

    std::cout << bwd_out << std::endl;
    std::cout << v1 << std::endl;
    std::cout << v2 << std::endl;


    return 0;
}
