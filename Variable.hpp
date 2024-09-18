#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>

#include "VariableImpl.hpp"


// Fancy wrapper for std::shared_pointer<VariableImpl<T>> for which the basic
// arithmetic operations are defined
template<class T>
class Variable {
public:
    Variable() {
        _variable = std::make_shared<VariableImpl<T>>();
    } 

    Variable(T value, bool requires_grad = false) {
        _variable = std::make_shared<VariableImpl<T>>(value, requires_grad, true);
    }

    Variable(T value, bool requires_grad, bool is_leaf, std::function<std::vector<T>(const T&)> backward_fn) {
        _variable = std::make_shared<VariableImpl<T>>(value, requires_grad, is_leaf, backward_fn);
    }

    T value() const { return _variable->value(); }
    T grad() const { return _variable->grad(); }
    bool requires_grad() const { return _variable->requires_grad(); }
    bool is_leaf() const { return _variable->is_leaf(); }

    const std::vector<std::shared_ptr<VariableImpl<T>>>& parents() const {
        return _variable->parents();
    }

    void backward(T prev_grad = 1, bool retain_graph = false) {
        _variable->backward(prev_grad, retain_graph);
    }

    Variable<T> exp() const {
        // Define the backward function for the exp operation
        std::function<std::vector<T>(const T&)> backward_fn = [this](const T& prev_grad) {
            std::vector<T> out_grads = {
                prev_grad * std::exp(this->value()),
            };
            return out_grads;
        };

        Variable<T> out(std::exp(this->value()), this->requires_grad(), false, backward_fn);
        out.add_parent(*this);
        return out;
    }

    Variable<T> log() const {
        // Define the backward function for the log operation
        std::function<std::vector<T>(const T&)> backward_fn = [this](const T& prev_grad) {
            std::vector<T> out_grads = {
                prev_grad * 1 / this->value(),
            };
            return out_grads;
        };

        Variable<T> out(std::log(this->value()), this->requires_grad(), false, backward_fn);
        out.add_parent(*this);
        return out;
    }

    Variable<T> sin() const {
        // Define the backward function for the sin operation
        std::function<std::vector<T>(const T&)> backward_fn = [this](const T& prev_grad) {
            std::vector<T> out_grads = {
                prev_grad * std::cos(this->value()),
            };
            return out_grads;
        };

        Variable<T> out(std::sin(this->value()), this->requires_grad(), false, backward_fn);
        out.add_parent(*this);
        return out;
    }

    Variable<T> cos() const {
        // Define the backward function for the cos operation
        std::function<std::vector<T>(const T&)> backward_fn = [this](const T& prev_grad) {
            std::vector<T> out_grads = {
                prev_grad * -std::sin(this->value()),
            };
            return out_grads;
        };

        Variable<T> out(std::cos(this->value()), this->requires_grad(), false, backward_fn);
        out.add_parent(*this);
        return out;
    }


    //  Define operations for all combinations (Variable op Variable, Variable op Generic, Generic op Variable)
    #define DEFINE_BINARY_OPERATORS(op) \
    /* Variable op Variable */ \
    template<class A> \
    friend Variable<A> operator op (const Variable<A>& lhs, const Variable<A>& rhs); \
    \
    /* Variable op Generic */ \
    template<class A> \
    friend Variable<A> operator op (const Variable<A>& lhs, const A& rhs); \
    \
    /* Generic op Variable */ \
    template<class A> \
    friend Variable<A> operator op (const A& lhs, const Variable<A>& rhs); \

    // Define the operations once
    DEFINE_BINARY_OPERATORS(+)
    DEFINE_BINARY_OPERATORS(-)
    DEFINE_BINARY_OPERATORS(*)
    DEFINE_BINARY_OPERATORS(/)

    #undef DEFINE_BINARY_OPERATORS

    template<class A>
    friend std::ostream& operator<<(std::ostream& os, Variable<A>& var);

    template<class A>
    friend Variable<A> operator+(const Variable<A>& lhs, const Variable<A>& rhs);

    std::shared_ptr<VariableImpl<T>> _variable; 

private:
       

    void add_parent(Variable parent) {
        _variable->add_parent(parent._variable);
    } 
};



template<class T>
std::ostream& operator<<(std::ostream& os, Variable<T>& var) {
    os << "Variable(" << var.value() << ", grad=" << var.grad() << ", requires_grad=" << var.requires_grad() << ")";
    os << " @ " << var._variable.get() << " | " << var._variable.use_count();
    if (var.requires_grad() & !var.parents().empty()) {
        os  << std::endl << "  Parents: [";
        for (const auto& parent : var.parents()) {
            os << parent->value()  << " (" << parent.get() << " | " << parent.use_count() << "), ";
        }
        os << "]";
    }
    return os;
}

template<class T>
Variable<T> operator+(const Variable<T>& lhs, const Variable<T>& rhs) {
    bool requires_grad = lhs.requires_grad() || rhs.requires_grad();

    // Define the backward function for the addition operation
    std::function<std::vector<T>(const T&)> backward_fn = [&lhs, &rhs](const T& prev_grad) {
        std::vector<T> out_grads = {
            prev_grad,
            prev_grad,
        };
        return out_grads;
    };

    Variable<T> out(lhs.value() + rhs.value(), requires_grad, false, backward_fn);
    out.add_parent(lhs);
    out.add_parent(rhs);

    return out;
}

template<class T>
Variable<T> operator+(const Variable<T>& lhs, const T& rhs) {
    return lhs + Variable<T>(rhs);
}

template<class T>
Variable<T> operator+(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs) + rhs;
}


template<class T>
Variable<T> operator-(const Variable<T>& lhs, const Variable<T>& rhs) {
    bool requires_grad = lhs.requires_grad() || rhs.requires_grad();

    // Define the backward function for the addition operation
    std::function<std::vector<T>(const T&)> backward_fn = [&lhs, &rhs](const T& prev_grad) {
        std::vector<T> out_grads = {
            prev_grad,
            -prev_grad,
        };
        return out_grads;
    };

    Variable<T> out(lhs.value() - rhs.value(), requires_grad, false, backward_fn);
    out.add_parent(lhs);
    out.add_parent(rhs);

    return out;
}

template<class T>
Variable<T> operator-(const Variable<T>& lhs, const T& rhs) {
    return lhs - Variable<T>(rhs);
}

template<class T>
Variable<T> operator-(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs) - rhs;
}


template<class T>
Variable<T> operator*(const Variable<T>& lhs, const Variable<T>& rhs) {
    bool requires_grad = lhs.requires_grad() || rhs.requires_grad();

    // Define the backward function for the addition operation
    std::function<std::vector<T>(const T&)> backward_fn = [&lhs, &rhs](const T& prev_grad) {
        std::vector<T> out_grads = {
            prev_grad * rhs.value(),
            prev_grad * lhs.value(),
        };
        return out_grads;
    };
    Variable<T> out(lhs.value() * rhs.value(), requires_grad, false, backward_fn);
    out.add_parent(lhs);
    out.add_parent(rhs);

    return out;
}

template<class T>
Variable<T> operator*(const Variable<T>& lhs, const T& rhs) {
    return lhs * Variable<T>(rhs);
}

template<class T>
Variable<T> operator*(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs) * rhs;
}

template<class T>
Variable<T> operator/(const Variable<T>& lhs, const Variable<T>& rhs) {
    bool requires_grad = lhs.requires_grad() || rhs.requires_grad();

    // Define the backward function for the addition operation
    std::function<std::vector<T>(const T&)> backward_fn = [&lhs, &rhs](const T& prev_grad) {
        std::vector<T> out_grads = {
            prev_grad * 1 / rhs.value(),
            prev_grad * lhs.value(),
        };
        return out_grads;
    };
    Variable<T> out(lhs.value() / rhs.value(), requires_grad, false, backward_fn);
    out.add_parent(lhs);
    out.add_parent(rhs);

    return out;
}

template<class T>
Variable<T> operator/(const Variable<T>& lhs, const T& rhs) {
    return lhs / Variable<T>(rhs);
}

template<class T>
Variable<T> operator/(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs) / rhs;
}