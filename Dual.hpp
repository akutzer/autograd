#pragma once
#include <iostream>
#include <type_traits>
#include <cmath>



template<typename T>
class Dual {
private:
    T _primal, _tangent;

public:
    Dual(T primal, T tangent = 0) : _primal(primal), _tangent(tangent) {};

    T primal() const { return _primal; }
    T tangent() const { return _tangent; }


    ///////////////////////////////////////////////////////////////////////////
    ///                          UNARY OPERATIONS                           ///
    ///////////////////////////////////////////////////////////////////////////

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

    template<typename A>
    friend Dual<A> operator-(const Dual<A>& val);

    template<typename U>
    friend std::ostream& operator<<(std::ostream& ios, const Dual<U>& dual);


    ///////////////////////////////////////////////////////////////////////////
    ///                          BINARY OPERATIONS                          ///
    ///////////////////////////////////////////////////////////////////////////

    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator+(const Dual<A>& lhs, const Dual<B>& rhs);
    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator+(const Dual<A>& lhs, const B& rhs);
    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator+(const A& lhs, const Dual<B>& rhs);

    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator-(const Dual<A>& lhs, const Dual<B>& rhs);
    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator-(const Dual<A>& lhs, const B& rhs);
    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator-(const A& lhs, const Dual<B>& rhs);

    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator*(const Dual<A>& lhs, const Dual<B>& rhs);
    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator*(const Dual<A>& lhs, const B& rhs);
    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator*(const A& lhs, const Dual<B>& rhs);

    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator/(const Dual<A>& lhs, const Dual<B>& rhs);
    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator/(const Dual<A>& lhs, const B& rhs);
    template<typename A, class B>
    friend Dual<typename std::common_type<A, B>::type> operator/(const A& lhs, const Dual<B>& rhs);

};



template<typename A>
Dual<A> operator-(const Dual<A>& val) {
    return Dual<A>(-val._primal, -val._tangent);
}



template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator+(const Dual<A>& lhs, const Dual<B>& rhs) {
    using X = typename std::common_type<A, B>::type;
    return Dual<X>(lhs._primal + rhs._primal, 1 * lhs._tangent + 1 * rhs._tangent);
}

template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator+(const Dual<A>& lhs, const B& rhs) {
    return lhs + Dual<B>(rhs, 0);
}

template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator+(const A& lhs, const Dual<B>& rhs) {
    return Dual<A>(lhs, 0) + rhs;
}



template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator-(const Dual<A>& lhs, const Dual<B>& rhs) {
    return lhs + rhs.negate();
}

template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator-(const Dual<A>& lhs, const B& rhs) {
    return lhs - Dual<B>(rhs, 0);
}

template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator-(const A& lhs, const Dual<B>& rhs) {
    return Dual<A>(lhs, 0) - rhs;
}



template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator*(const Dual<A>& lhs, const Dual<B>& rhs) {
    using X = typename std::common_type<A, B>::type;
    // Dual(a * b, da/dx * b + a * db/dx)
    return Dual<X>(lhs._primal * rhs._primal, lhs._tangent * rhs._primal + lhs._primal * rhs._tangent);
}

template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator*(const Dual<A>& lhs, const B& rhs) {
    return lhs * Dual<B>(rhs, 0);
}

template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator*(const A& lhs, const Dual<B>& rhs) {
    return Dual<A>(lhs, 0) * rhs;
}



template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator/(const Dual<A>& lhs, const Dual<B>& rhs) {
    using X = typename std::common_type<A, B>::type;
    // Dual(a / b, (da/dx * b - a * db/dx) / b²)
    return Dual<X>(lhs._primal / rhs._primal, (lhs._tangent * rhs._primal - lhs._primal * rhs._tangent) / (rhs._primal * rhs._primal));  
    // Dual(a * 1/b, (da/dx * 1/b - a * 1/b² * db/dx))
    // return lhs * rhs.reciprocal();
}

template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator/(const Dual<A>& lhs, const B& rhs) {
    return lhs / Dual<B>(rhs, 0);
}

template<typename A, class B>
Dual<typename std::common_type<A, B>::type> operator/(const A& lhs, const Dual<B>& rhs) {
    return Dual<A>(lhs, 0) / rhs;
}


///////////////////////////////////////////////////////////////////////////
///                              PRINTING                               ///
///////////////////////////////////////////////////////////////////////////

template<typename T>
struct std::formatter<Dual<T>> : std::formatter<std::string> {
    auto format(const Dual<T>& dual, format_context& ctx) const {
        return formatter<string>::format(
            std::format("Dual({:.12}, {:.12})", dual.primal(), dual.tangent()), ctx
        );
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Dual<T>& dual) {
    os << std::format("{}", dual);
    return os;
}
