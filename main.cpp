#include <iostream>
#include "Variable.hpp"
#include "Dual.hpp"



int main(int argc, char const *argv[])
{

    // Dual<float> x1(2, 1), x2(5, 0);
    // auto fwd_out = x1.log() + x1 * x2 - x2.sin();
    // std::cout << fwd_out << std::endl;

    // Dual<float> xx1(2, 0), xx2(5, 1);
    // auto fwd_out2 = xx1.log() + xx1 * xx2 - xx2.sin();
    // std::cout << fwd_out2 << std::endl;



    Variable<float> v1(2, true), v2(5, true);

    auto vv1 = v1.log();
    auto vv2 = v1 * v2;
    auto vv3 = v2.sin();
    auto vv4 = vv1 + vv2;
    auto vv5 = vv4 - vv3;
    
    vv5.backward(1, false);
    std::cout << vv5 << std::endl;
    std::cout << vv4 << std::endl;
    std::cout << vv3 << std::endl;
    std::cout << vv2 << std::endl;
    std::cout << vv1 << std::endl;
    std::cout << v1 << std::endl;
    std::cout << v2 << std::endl;


    // auto bwd_out = v1.log() + v1 * v2 - v2.sin();
    // // auto bwd_out = -v1;
    // bwd_out.backward(1, true);

    // std::cout << bwd_out << std::endl;
    // std::cout << v1 << std::endl;
    // std::cout << v2 << std::endl;

        // Variable<float> a(2, true), b(5, true);
    // Variable<float> c = a * b;
    // Variable<float> d = c + 1.f;
    // std::cout << "A" << std::endl;
    // d.backward(1, true);
    // std::cout << d << std::endl;
    // // std::cout << c << std::endl;
    // // std::cout << c.is_leaf() << std::endl;
    // std::cout << a << std::endl;
    // std::cout << b << std::endl;






    // Variable<float> a(2, true);
    // // auto b = -a;
    // // b.backward();
    // // std::cout << b << std::endl;
    // // std::cout << a << std::endl;
    
    // auto b = a + 1.f;

    // // Variable<float> x(3, false);
    

    // auto c = b * 3.f;
    // auto d = b + a;

    // auto e = c * d;

    // e.backward(1, true);
    
    // std::cout << e << std::endl;
    // std::cout << d << std::endl;
    // std::cout << c << std::endl;
    // std::cout << b << std::endl;
    // std::cout << a << std::endl;


    // Dual<float> a(2, 1);
    // auto b = a + 1.f;
    // auto c = b * 3.f;
    // auto d = b + a;
    // auto e = c * d;
    // std::cout << e << std::endl;



    return 0;
}
