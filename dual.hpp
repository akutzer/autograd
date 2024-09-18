#pragma once
#include <iostream>
#include <type_traits>
#include<cmath>


template<class T>
class Dual {
    private:
        T _primal, _tangent;

    public:
        Dual(T primal, T tangent = 0) : _primal(primal), _tangent(tangent) {};


        using type = T;

        Dual<typename std::common_type<T, decltype(-std::declval<T>())>::type> negate() const {
            using PromotedType = typename std::common_type<T, decltype(-std::declval<T>())>::type;
            PromotedType p = static_cast<PromotedType>(_primal);
            PromotedType t = static_cast<PromotedType>(_tangent);

            return Dual<PromotedType>(-p, -t);
        }

        Dual<typename std::common_type<T, decltype(-1. / std::declval<T>())>::type> reciprocal() const {
            using PromotedType = typename std::common_type<T, decltype(-1. / std::declval<T>())>::type;
            PromotedType p = static_cast<PromotedType>(_primal);
            PromotedType t = static_cast<PromotedType>(_tangent);

            return Dual<PromotedType>(1 / p, -1 / (p * p) * t);
        }

        Dual<T> log() const {
            return Dual<T>(std::log(_primal), 1 / _primal * _tangent);
        }

        Dual<T> exp() const {
            return Dual<T>(std::exp(_primal), std::exp(_primal) * _tangent);
        }

        Dual<T> sin() const {
            return Dual<T>(std::sin(_primal), std::cos(_primal) * _tangent);
        }

        Dual<T> cos() const {
            return Dual<T>(std::cos(_primal), -std::sin(_primal) * _tangent);
        }

        template<class U>
        friend std::ostream& operator<<(std::ostream& ios, const Dual<U>& dual);

        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator+(const Dual<A>& lhs, const Dual<B>& rhs);
        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator+(const Dual<A>& lhs, const B& rhs);
        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator+(const A& lhs, const Dual<B>& rhs);

        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator-(const Dual<A>& lhs, const Dual<B>& rhs);
        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator-(const Dual<A>& lhs, const B& rhs);
        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator-(const A& lhs, const Dual<B>& rhs);

        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator*(const Dual<A>& lhs, const Dual<B>& rhs);
        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator*(const Dual<A>& lhs, const B& rhs);
        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator*(const A& lhs, const Dual<B>& rhs);

        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator/(const Dual<A>& lhs, const Dual<B>& rhs);
        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator/(const Dual<A>& lhs, const B& rhs);
        template<class A, class B>
        friend Dual<typename std::common_type<A, B>::type> operator/(const A& lhs, const Dual<B>& rhs);



};

template<class T>
std::ostream& operator<<(std::ostream& os, const Dual<T>& dual) {
    os << "Dual(" << dual._primal << ", " << dual._tangent << ")";
    return os;
}


template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator+(const Dual<A>& lhs, const Dual<B>& rhs) {
    using X = typename std::common_type<A, B>::type;

    return Dual<X>(lhs._primal + rhs._primal, 1 * lhs._tangent + 1 * rhs._tangent);
}

template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator+(const Dual<A>& lhs, const B& rhs) {
    return lhs + Dual<B>(rhs, 0);
}

template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator+(const A& lhs, const Dual<B>& rhs) {
    return Dual<A>(lhs, 0) + rhs;
}


template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator-(const Dual<A>& lhs, const Dual<B>& rhs) {
    return lhs + rhs.negate();
}

template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator-(const Dual<A>& lhs, const B& rhs) {
    return lhs - Dual<B>(rhs, 0);
}

template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator-(const A& lhs, const Dual<B>& rhs) {
    return Dual<A>(lhs, 0) - rhs;
}


template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator*(const Dual<A>& lhs, const Dual<B>& rhs) {
    using X = typename std::common_type<A, B>::type;
    // Dual(a * b, da/dx * b + a * db/dx)
    return Dual<X>(lhs._primal * rhs._primal, lhs._tangent * rhs._primal + lhs._primal * rhs._tangent);
}

template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator*(const Dual<A>& lhs, const B& rhs) {
    return lhs * Dual<B>(rhs, 0);
}

template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator*(const A& lhs, const Dual<B>& rhs) {
    return Dual<A>(lhs, 0) * rhs;
}


template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator/(const Dual<A>& lhs, const Dual<B>& rhs) {
    // Dual(a / b, (da/dx * b - a * db/dx) / b²)
    // Dual(a * 1/b, (da/dx * 1/b - a * 1/b² * db/dx))
    // using X = typename std::common_type<A, B>::type;
    // return Dual<X>(lhs._primal / rhs._primal, (lhs._tangent * rhs._primal - lhs._primal * rhs._tangent) / (rhs._primal * rhs._primal));
    return lhs * rhs.reciprocal();
}

template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator/(const Dual<A>& lhs, const B& rhs) {
    return lhs / Dual<B>(rhs, 0);
}

template<class A, class B>
Dual<typename std::common_type<A, B>::type> operator/(const A& lhs, const Dual<B>& rhs) {
    return Dual<A>(lhs, 0) / rhs;
}