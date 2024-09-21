#include <iostream>
#include <iomanip>
#include <print>
#include "Variable.hpp"
#include "Dual.hpp"


template<typename T>
T f(const T& x, const T& y) {
    auto tmp = (x.log() + (-x) * y - y.sin());
    if ((tmp * 2.f).value() < 0) {
        tmp = tmp * tmp;
    }
    auto tmp2 = tmp;
    for (int i = 1; i < 5; ++i)
        tmp = tmp * ((y - x) / static_cast<T>(i)).exp();
    auto tmp3 = -(tmp * 5.f).sin(); // unused variable
    return tmp / ((static_cast<T>(2) * x).cos().abs() + tmp2);
}



int main(int argc, char const *argv[])
{
    using dtype = float;

    // Variable<dtype> X(0, 1);
    // Variable<dtype> Y;
    // Variable<dtype> D;
    // {
    //     auto A = X + 1.f;
    //     auto B = A + 2.f;
    //     auto C = A + 4.f;
    //     D = B * C;
    //     Y = B * 1.4f;
    //     std::println("{:d}", Y);
    //     // std::cout << A << std::endl;
    // }
    // // auto D = (A + 2.f) * (A + 4.f);
    // D.backward();
    // std::println("{:d}", Y);
    // std::println("{:d}", D);
    // std::println("{:d}", X);

    // return 0;

    std::println("{:~^50}", " Forward mode differentiation: ");
    // forward mode differentiation needs two calls, first to compute
    // df/ddual_x and then df/ddual_y
    Dual<dtype> dual_x(2, 1), dual_y(5, 0);
    auto dual_out = f(dual_x, dual_y);
    std::println("{}", dual_out);

    dual_x = Dual<dtype>(2, 0);
    dual_y = Dual<dtype>(5, 1);
    dual_out = f(dual_x, dual_y);
    std::println("{}", dual_out);


    std::println("\n\n{:~^50}", " Backward mode differentiation: ");
    // backwards mode differentiation needs a forward and then backward call to
    // compute df/dx and then df/dy
    Variable<dtype> x(2, true), y(5, true);
    auto out = f(x, y);
    out.backward();
    std::println("{:d}", out); // print in debug mode
    std::println("{}", x);
    std::println("{}", y);

   
    std::println("\n\nOut-of-scope variables are kept alive if they are part of the final computation graph:");
    Variable<dtype> z;
    {
        Variable<dtype> tmp(5, true);
        std::cout << "tmp._variable address: " << tmp.variable() << std::endl;
        z = tmp * 2.f;
    }
    std::println("{:d}", z); // z keeps tmp._variable alive since it has a shared pointer to it


    std::println("\n\nOut-of-scope variables that are not part of the final computation graph are not kept alive:");
    Variable<dtype> xx(5, true);
    {
        Variable<dtype> yy(2, true);
        std::cout << "yy._variable address: " << yy.variable() << std::endl;
        yy = xx * 2.f;
    }
    std::println("Use counts of the children of xx:");
    for (const auto& child: xx.variable()->children()) {
        std::cout << child.use_count() << std::endl; // xx keeps yy._variable not alive since it has only a weak pointer to it
    }
    // std::println("{:d}", xx);


    std::println("\n\n{:~^50}", " Forward mode differentiation: ");
    Dual<dtype> aa(2, 1), bb(5, 0);
    auto fwd_out = aa.log() + aa * bb - bb.sin();
    std::println("{}", fwd_out);

    aa = Dual<dtype>(2, 0);
    bb = Dual<dtype>(5, 1);
    fwd_out = aa.log() + aa * bb - bb.sin();
    std::println("{}", fwd_out);

    std::println("\n\n{:~^50}", " Backward mode differentiation: ");
    Variable<dtype> a(2, true), b(5, true);
    auto bwd_out = a.log() + a * b - b.sin();

    std::println("First call:");
    bwd_out.backward(1, true); // retain the computation graph for a second call
    std::println("{:d}", bwd_out);
    std::println("{:d}", a);
    std::println("{:d}", b);
    
    std::println("\nSecond call:");
    // call backwards a second time, but this time clear the computation graph
    bwd_out.backward(1, false);
    std::println("{:d}", bwd_out);
    std::println("{:d}", a);
    std::println("{:d}", b);

    return 0;
}
