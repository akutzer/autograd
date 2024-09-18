#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>

#include "VariableImpl.hpp"


namespace GradRegistry {

    template<class T>
    T neg_fwd(const T val) {
        return -val;
    }

    template<class T>
    std::vector<T> neg_bwd(const VariableImpl<T>& val, const T& prev_grad) {
        return {-prev_grad};
    }

    template<class T>
    T add_fwd(const T lhs, const T rhs) {
        return lhs + rhs;
    }

    template<class T>
    std::vector<T> add_bwd(const VariableImpl<T>& lhs, const VariableImpl<T>& rhs, const T& prev_grad) {
        return {prev_grad, prev_grad};
    }

    template<class T>
    T sub_fwd(const T lhs, const T rhs) {
        return lhs - rhs;
    }

    template<class T>
    std::vector<T> sub_bwd(const VariableImpl<T>& lhs, const VariableImpl<T>& rhs, const T& prev_grad) {
        return {prev_grad, -prev_grad};
    }

    template<class T>
    T mul_fwd(const T lhs, const T rhs) {
        return lhs * rhs;
    }

    template<class T>
    std::vector<T> mul_bwd(const VariableImpl<T>& lhs, const VariableImpl<T>& rhs, const T& prev_grad) {
        return {prev_grad * rhs.value(), prev_grad * lhs.value()};
    }

    template<class T>
    T div_fwd(const T lhs, const T rhs) {
        return lhs / rhs;
    }

    template<class T>
    std::vector<T> div_bwd(const VariableImpl<T>& lhs, const VariableImpl<T>& rhs, const T& prev_grad) {
        return {prev_grad * 1 / rhs.value(), prev_grad * -lhs.value() / (rhs.value() * rhs.value())};
    }

    template<class T>
    T exp_fwd(T val) {
        return std::exp(val);
    }

    template<class T>
    std::vector<T> exp_bwd(const VariableImpl<T>& var, const T& prev_grad) {
        return {prev_grad * std::exp(var.value())};
    }

    template<class T>
    T log_fwd(T val) {
        return std::log(val);
    }

    template<class T>
    std::vector<T> log_bwd(const VariableImpl<T>& var, const T& prev_grad) {
        return {prev_grad * 1 / var.value()};
    }

    template<class T>
    T sin_fwd(T val) {
        return std::sin(val);
    }

    template<class T>
    std::vector<T> sin_bwd(const VariableImpl<T>& var, const T& prev_grad) {
        return {prev_grad * std::cos(var.value())};
    }

    template<class T>
    T cos_fwd(T val) {
        return std::cos(val);
    }

    template<class T>
    std::vector<T> cos_bwd(const VariableImpl<T>& var, const T& prev_grad) {
        return {prev_grad * -std::sin(var.value())};
    }
}


// Fancy wrapper for std::shared_pointer<VariableImpl<T>> for which the basic
// arithmetic operations are defined
template<class T>
class Variable {
public:
    // user created Variables are by default `leaf`
    Variable(T value, bool requires_grad = false, bool is_leaf = true)
        : _variable(std::make_shared<VariableImpl<T>>(value, requires_grad, is_leaf)) {}
    
    // copy & copy-assign constructor which increment the reference count
    // of th underlying VariableImpl
    Variable(const Variable<T>& other) : _variable(other._variable) {
        if (_variable)
            _variable->increment_ref_count();
    }

    Variable<T>& operator=(const Variable<T>& other) {
        if (this != &other) {
            _variable = other._variable;
            if (_variable)
                _variable->increment_ref_count();
        }
        return *this;
    }

    Variable(Variable<T>&& other) noexcept : _variable(std::move(other._variable)) {}

    Variable<T>& operator=(Variable<T>&& other) noexcept {
        if (this != &other) {
            _variable = std::move(other._variable);
        }
        return *this;
    }

    T value() const { return _variable->value(); }
    std::optional<T> grad() const { return _variable->grad(); }
    bool requires_grad() const { return _variable->requires_grad(); }
    bool is_leaf() const { return _variable->is_leaf(); }

    void backward(T prev_grad = 1, bool retain_graph = false) {
        _variable->backward(prev_grad, retain_graph);
    }

    const std::vector<std::shared_ptr<VariableImpl<T>>>& parents() const {
        return _variable->parents();
    }


    ///////////////////////////////////////////////////////////////////////////
    ///                          UNARY OPERATIONS                           ///
    ///////////////////////////////////////////////////////////////////////////

    template<class A>
    friend Variable<A> operator-(const Variable<A>& var);

    Variable<T> exp() const {
        Variable<T> out(GradRegistry::exp_fwd(this->value()), this->requires_grad());
        out._variable->set_backward_fn([this](const T& prev_grad) {
            return GradRegistry::exp_bwd(*this->_variable, prev_grad);
        });
        out._variable->add_parent(_variable);
        return out;
    }

    Variable<T> log() const {
        Variable<T> out(GradRegistry::log_fwd(this->value()), this->requires_grad());
        out._variable->set_backward_fn([this](const T& prev_grad) {
            return GradRegistry::log_bwd(*this->_variable, prev_grad);
        });
        out._variable->add_parent(_variable);
        return out;
    }

    Variable<T> sin() const {
        Variable<T> out(GradRegistry::sin_fwd(this->value()), this->requires_grad());
        out._variable->set_backward_fn([this](const T& prev_grad) {
            return GradRegistry::sin_bwd(*this->_variable, prev_grad);
        });
        out._variable->add_parent(_variable);
        return out;
    }

    Variable<T> cos() const {
        Variable<T> out(GradRegistry::cos_fwd(this->value()), this->requires_grad());
        out._variable->set_backward_fn([this](const T& prev_grad) {
            return GradRegistry::cos_bwd(*this->_variable, prev_grad);
        });
        out._variable->add_parent(_variable);
        return out;
    }


    ///////////////////////////////////////////////////////////////////////////
    ///                          BINARY OPERATIONS                          ///
    ///////////////////////////////////////////////////////////////////////////

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

private:
    std::shared_ptr<VariableImpl<T>> _variable;

};


template<class T>
Variable<T> operator-(const Variable<T>& var) {
    Variable<T> out(GradRegistry::neg_fwd(var.value()), var.requires_grad(), false);
    out._variable->set_backward_fn([var = var._variable](const T& prev_grad) {
        return GradRegistry::neg_bwd(*var, prev_grad);
    });
    out._variable->add_parent(var._variable);
    return out;
}


template<class T>
Variable<T> operator+(const Variable<T>& lhs, const Variable<T>& rhs) {
    Variable<T> out(GradRegistry::add_fwd(lhs.value(), rhs.value()), lhs.requires_grad() || rhs.requires_grad(), false);
    out._variable->set_backward_fn([lhs_var = lhs._variable, rhs_var = rhs._variable](const T& prev_grad) {
        return GradRegistry::add_bwd(*lhs_var, *rhs_var, prev_grad);
    });
    out._variable->add_parent(lhs._variable);
    out._variable->add_parent(rhs._variable);
    return out;
}

template<class T>
Variable<T> operator+(const Variable<T>& lhs, const T& rhs) {
    Variable<T> out = lhs + Variable<T>(rhs, false, false);
    return out;
}

template<class T>
Variable<T> operator+(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs, false, false) + rhs;
}


template<class T>
Variable<T> operator-(const Variable<T>& lhs, const Variable<T>& rhs) {
    Variable<T> out(GradRegistry::sub_fwd(lhs.value(), rhs.value()), lhs.requires_grad() || rhs.requires_grad(), false);
    out._variable->set_backward_fn([lhs_var = lhs._variable, rhs_var = rhs._variable](const T& prev_grad) {
        return GradRegistry::sub_bwd(*lhs_var, *rhs_var, prev_grad);
    });
    out._variable->add_parent(lhs._variable);
    out._variable->add_parent(rhs._variable);
    return out;
}

template<class T>
Variable<T> operator-(const Variable<T>& lhs, const T& rhs) {
    return lhs - Variable<T>(rhs, false, false);
}

template<class T>
Variable<T> operator-(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs, false, false) - rhs;
}


template<class T>
Variable<T> operator*(const Variable<T>& lhs, const Variable<T>& rhs) {
    Variable<T> out(GradRegistry::mul_fwd(lhs.value(), rhs.value()), lhs.requires_grad() || rhs.requires_grad(), false);
    out._variable->set_backward_fn([lhs_var = lhs._variable, rhs_var = rhs._variable](const T& prev_grad) {
        return GradRegistry::mul_bwd(*lhs_var, *rhs_var, prev_grad);
    });
    out._variable->add_parent(lhs._variable);
    out._variable->add_parent(rhs._variable);
    return out;
}

template<class T>
Variable<T> operator*(const Variable<T>& lhs, const T& rhs) {
    return lhs * Variable<T>(rhs, false, false);
}

template<class T>
Variable<T> operator*(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs, false, false) * rhs;
}


template<class T>
Variable<T> operator/(const Variable<T>& lhs, const Variable<T>& rhs) {
    Variable<T> out(GradRegistry::div_fwd(lhs.value(), rhs.value()), lhs.requires_grad() || rhs.requires_grad(), false);
    out._variable->set_backward_fn([lhs_var = lhs._variable, rhs_var = rhs._variable](const T& prev_grad) {
        return GradRegistry::div_bwd(*lhs_var, *rhs_var, prev_grad);
    });
    out._variable->add_parent(lhs._variable);
    out._variable->add_parent(rhs._variable);
    return out;
}

template<class T>
Variable<T> operator/(const Variable<T>& lhs, const T& rhs) {
    return lhs / Variable<T>(rhs, false, false);
}

template<class T>
Variable<T> operator/(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs, false, false) / rhs;
}


template<class T>
std::ostream& operator<<(std::ostream& os, Variable<T>& var) {
    os << "Variable(" << var.value();
    if (var.grad())
        os << ", grad=" << var.grad().value();

    if (var.requires_grad())
        os << ", requires_grad=" << var.requires_grad();
    
    os << ")";

    if (var.requires_grad() && !var.parents().empty()) {
        os  << std::endl << " └─ Parents: [";
        for (const auto& parent : var.parents()) {
            os << parent->value()  << " (" << parent.get() << " | " << parent.use_count() << "), ";
        }
        os << "]";
    }
    return os;
}
