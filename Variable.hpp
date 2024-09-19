#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>

#include "VariableImpl.hpp"




namespace GradRegistry {

    ///////////////////////////////////////////////////////////////////////////
    ///                          BINARY OPERATIONS                          ///
    ///////////////////////////////////////////////////////////////////////////

    struct Add {
        template<typename T>
        T operator()(const T lhs, const T rhs) const { return lhs + rhs; }

        template<typename T>
        std::vector<T> backward(const VariableImpl<T>& lhs, const VariableImpl<T>& rhs, const T& prev_grad) const {
            return {prev_grad, prev_grad};
        }
    };

    struct Sub {
        template<typename T>
        T operator()(const T lhs, const T rhs) const { return lhs - rhs; }

        template<typename T>
        std::vector<T> backward(const VariableImpl<T>& lhs, const VariableImpl<T>& rhs, const T& prev_grad) const {
            return {prev_grad, -prev_grad};
        }
    };

    struct Mul {
        template<typename T>
        T operator()(const T lhs, const T rhs) const { return lhs * rhs; }

        template<typename T>
        std::vector<T> backward(const VariableImpl<T>& lhs, const VariableImpl<T>& rhs, const T& prev_grad) const {
            return {prev_grad * rhs.value(), prev_grad * lhs.value()};
        }
    };

    struct Div {
        template<typename T>
        T operator()(const T lhs, const T rhs) const { return lhs / rhs; }

        template<typename T>
        std::vector<T> backward(const VariableImpl<T>& lhs, const VariableImpl<T>& rhs, const T& prev_grad) const {
            return {prev_grad * 1 / rhs.value(), prev_grad * -lhs.value() / (rhs.value() * rhs.value())};
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    ///                          UNARY OPERATIONS                           ///
    ///////////////////////////////////////////////////////////////////////////

    struct Neg {
        template<typename T>
        T operator()(const T val) const { return -val; }

        template<typename T>
        std::vector<T> backward(const VariableImpl<T>& var, const T& prev_grad) const {
            return {-prev_grad};
        }
    };

    struct Reciprocal {
        template<typename T>
        T operator()(const T val) const { return 1/val; }

        template<typename T>
        std::vector<T> backward(const VariableImpl<T>& var, const T& prev_grad) const {
            return {prev_grad * -1 / (var.value() * var.value()) };
        }
    };

    struct Exp {
        template<typename T>
        T operator()(const T val) const { return std::exp(val); }

        template<typename T>
        std::vector<T> backward(const VariableImpl<T>& var, const T& prev_grad) const {
            return {prev_grad * std::exp(var.value())};
        }
    };

    struct Log {
        template<typename T>
        T operator()(const T val) const { return std::log(val); }

        template<typename T>
        std::vector<T> backward(const VariableImpl<T>& var, const T& prev_grad) const {
            return {prev_grad * 1 / var.value()};
        }
    };

    struct Sin {
        template<typename T>
        T operator()(const T val) const { return std::sin(val); }

        template<typename T>
        std::vector<T> backward(const VariableImpl<T>& var, const T& prev_grad) const {
            return {prev_grad * std::cos(var.value())};
        }
    };

    struct Cos {
        template<typename T>
        T operator()(const T val) const { return std::cos(val); }

        template<typename T>
        std::vector<T> backward(const VariableImpl<T>& var, const T& prev_grad) const {
            return {prev_grad * -std::sin(var.value())};
        }
    };
}


// Fancy wrapper for std::shared_pointer<VariableImpl<T>> for which the basic
// arithmetic operations are defined
template<typename T>
class Variable {
public:
    // user created Variables are by default `leaf`
    Variable(T value, bool requires_grad = false, bool is_leaf = true)
        : _variable(std::make_shared<VariableImpl<T>>(value, requires_grad, is_leaf)) {
            // _variable->increment_num_children();
        }
    
    // copy & copy-assign constructors perform a shallow copy, with the new
    // this referencing the same VariableImpl as other (i.e. copy returns
    // a view of the underlying VariabelImpl)
    Variable(const Variable<T>& other) : _variable(other._variable) {}

    Variable<T>& operator=(const Variable<T>& other) {
        if (this != &other) {
            _variable = other._variable;
        }
        return *this;
    }

    Variable(Variable<T>&& other) noexcept : _variable(std::move(other._variable)) {
    }

    Variable<T>& operator=(Variable<T>&& other) noexcept {
        if (this != &other) {
            _variable = std::move(other._variable);
        }
        return *this;
    }

    T value() const { return _variable->value(); }
    std::optional<T> grad() const { return _variable->grad(); }
    void zero_grad() { _variable->zero_grad(); }
    bool requires_grad() const { return _variable->requires_grad(); }
    bool set_requires_grad(bool v = true) { return _variable->set_requires_grad(v); }
    bool is_leaf() const { return _variable->is_leaf(); }

    void backward(T prev_grad = 1, bool retain_graph = false) {
        _variable->backward(prev_grad, retain_graph);
    }

    const std::vector<std::shared_ptr<VariableImpl<T>>>& parents() const {
        return _variable->parents();
    }

    template<typename A, typename Op>
    friend Variable<A> binary_operation(const Variable<A>& lhs, const Variable<A>& rhs, const Op& op);

    template<typename A, typename Op>
    friend Variable<A> unary_operation(const Variable<A>& var, const Op& op);


    ///////////////////////////////////////////////////////////////////////////
    ///                          UNARY OPERATIONS                           ///
    ///////////////////////////////////////////////////////////////////////////

    template<typename A>
    friend Variable<A> operator-(const Variable<A>& var);

    Variable<T> exp() const {
        return unary_operation(*this, GradRegistry::Exp{});
    }

    Variable<T> log() const {
        return unary_operation(*this, GradRegistry::Log{});
    }

    Variable<T> sin() const {
        return unary_operation(*this, GradRegistry::Sin{});
    }

    Variable<T> cos() const {
        return unary_operation(*this, GradRegistry::Cos{});
    }


    ///////////////////////////////////////////////////////////////////////////
    ///                          BINARY OPERATIONS                          ///
    ///////////////////////////////////////////////////////////////////////////

    template<typename A>
    friend Variable<A> operator+(const Variable<A>& lhs, const Variable<A>& rhs);
    template<typename A>
    friend Variable<A> operator+(const Variable<A>& lhs, const A& rhs);
    template<typename A>
    friend Variable<A> operator+(const A& lhs, const Variable<A>& rhs);

    template<typename A>
    friend Variable<A> operator-(const Variable<A>& lhs, const Variable<A>& rhs);
    template<typename A>
    friend Variable<A> operator-(const Variable<A>& lhs, const A& rhs);
    template<typename A>
    friend Variable<A> operator-(const A& lhs, const Variable<A>& rhs);

    template<typename A>
    friend Variable<A> operator*(const Variable<A>& lhs, const Variable<A>& rhs);
    template<typename A>
    friend Variable<A> operator*(const Variable<A>& lhs, const A& rhs);
    template<typename A>
    friend Variable<A> operator*(const A& lhs, const Variable<A>& rhs);

    template<typename A>
    friend Variable<A> operator/(const Variable<A>& lhs, const Variable<A>& rhs);
    template<typename A>
    friend Variable<A> operator/(const Variable<A>& lhs, const A& rhs);
    template<typename A>
    friend Variable<A> operator/(const A& lhs, const Variable<A>& rhs);


    template<typename A>
    friend std::ostream& operator<<(std::ostream& os, const Variable<A>& var);

    std::shared_ptr<VariableImpl<T>> _variable;

private:
    // std::shared_ptr<VariableImpl<T>> _variable;

};


template<typename T, typename Op>
Variable<T> binary_operation(const Variable<T>& lhs, const Variable<T>& rhs, const Op& op) {
    Variable<T> out(op(lhs.value(), rhs.value()), lhs.requires_grad() || rhs.requires_grad(), false);
    // Variables created by operations are non-leaf

    if (out.requires_grad()) {
        // the only way lhs_wp or rhs_wp are invalid is when this->_variable
        // has deleted its parents, but this is only done in this->_variable->backward()
        // when the reference count is 0 and then _backward_fn is also set to
        // nullptr, thus _backward_fn can no longer be called in
        // this->_variable->backward()
        out._variable->set_backward_fn([lhs_wp = std::weak_ptr<VariableImpl<T>>(lhs._variable),
                                        rhs_wp = std::weak_ptr<VariableImpl<T>>(rhs._variable), &op](const T& prev_grad) {
            auto lhs_var = lhs_wp.lock();
            auto rhs_var = rhs_wp.lock();
            if (lhs_var && rhs_var) {
                return op.backward(*lhs_var, *rhs_var, prev_grad);
            }
            return std::vector<T>{};
        });

        out._variable->add_parent(lhs._variable);
        out._variable->add_parent(rhs._variable);
    }

    return out;
}

template<typename T, typename Op>
Variable<T> unary_operation(const Variable<T>& var, const Op& op) {
    Variable<T> out(op(var.value()), var.requires_grad(), false);
    // Variables created by operations are non-leaf

    if (out.requires_grad()) {
        out._variable->set_backward_fn([var_wp = std::weak_ptr<VariableImpl<T>>(var._variable), &op](const T& prev_grad) {
            auto var = var_wp.lock();
            if (var)
                return op.backward(*var, prev_grad);
            return std::vector<T>{};
        });

        out._variable->add_parent(var._variable);
    }

    return out;
}


template<typename T>
Variable<T> operator-(const Variable<T>& var) {
    return unary_operation(var, GradRegistry::Neg{});
}


template <typename T>
Variable<T> operator+(const Variable<T>& lhs, const Variable<T>& rhs) {
    return binary_operation(lhs, rhs, GradRegistry::Add{});
}

template<typename T>
Variable<T> operator+(const Variable<T>& lhs, const T& rhs) {
    Variable<T> out = lhs + Variable<T>(rhs, false, false);
    return out;
}

template<typename T>
Variable<T> operator+(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs, false, false) + rhs;
}


template<typename T>
Variable<T> operator-(const Variable<T>& lhs, const Variable<T>& rhs) {
    return binary_operation(lhs, rhs, GradRegistry::Sub{});
}

template<typename T>
Variable<T> operator-(const Variable<T>& lhs, const T& rhs) {
    return lhs - Variable<T>(rhs, false, false);
}

template<typename T>
Variable<T> operator-(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs, false, false) - rhs;
}


template<typename T>
Variable<T> operator*(const Variable<T>& lhs, const Variable<T>& rhs) {
    return binary_operation(lhs, rhs, GradRegistry::Mul{});
}

template<typename T>
Variable<T> operator*(const Variable<T>& lhs, const T& rhs) {
    return lhs * Variable<T>(rhs, false, false);
}

template<typename T>
Variable<T> operator*(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs, false, false) * rhs;
}


template<typename T>
Variable<T> operator/(const Variable<T>& lhs, const Variable<T>& rhs) {
    return binary_operation(lhs, rhs, GradRegistry::Div{});
}

template<typename T>
Variable<T> operator/(const Variable<T>& lhs, const T& rhs) {
    return lhs / Variable<T>(rhs, false, false);
}

template<typename T>
Variable<T> operator/(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs, false, false) / rhs;
}


template<typename T>
std::ostream& operator<<(std::ostream& os, const Variable<T>& var) {
    os << "Variable(" << var.value();
    if (var.grad())
        os << ", grad=" << var.grad().value();

    if (var.requires_grad())
        os << ", requires_grad=" << var.requires_grad();
    
    os << ")" << " use_count: " << var._variable.use_count();

    if (var.requires_grad() && !var.parents().empty()) {
        os  << std::endl << " └─ Parents: [";
        for (const auto& parent : var.parents()) {
            os << parent->value()  << " (" << parent.get() << " | " << parent.use_count() << "), ";
        }
        os << "]";
    }
    return os;
}
