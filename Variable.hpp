#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>

#include "VariableImpl.hpp"




namespace OperatorRegistry {

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


// The Variable class has two purposes:
// (1) it wraps a std::shared_pointer<VariableImpl<T>>
// (2) defines basic arithmetic operations for these shared pointers as well as
//     registers the backward function, which maps the grad w.r.t. the output
//     to the grad w.r.t. the input
template<typename T>
class Variable {
public:
    Variable() = default;

    // user created Variables are by default `leaf`
    Variable(T value, bool requires_grad = false, bool is_leaf = true)
        : _variable(std::make_shared<VariableImpl<T>>(value, requires_grad, is_leaf)) {}
    
    // copy & copy-assign constructors perform a shallow copy, meaning
    // this._variable = other._variable
    // (i.e. copy returns a view of the underlying VariableImpl)
    Variable(const Variable<T>& other) : _variable(other._variable) {}

    Variable<T>& operator=(const Variable<T>& other) {
        if (this != &other) {
            _variable = other._variable;
        }
        return *this;
    }

    // move & move-assign constructors
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
    const std::shared_ptr<VariableImpl<T>>& variable() const { return _variable; }

    void backward(T prev_grad = 1, bool retain_graph = false) {
        _variable->backward(prev_grad, retain_graph, nullptr, _variable);
    }

    const std::vector<std::shared_ptr<VariableImpl<T>>>& parents() const {
        return _variable->parents();
    }

    const std::vector<std::weak_ptr<VariableImpl<T>>>& children() const {
        return _variable->children();
    }

    template<typename A>
    friend std::ostream& operator<<(std::ostream& os, const Variable<A>& var);


    ///////////////////////////////////////////////////////////////////////////
    ///                         TEMPLATE OPERATIONS                         ///
    ///////////////////////////////////////////////////////////////////////////

    template<typename A, typename Op>
    friend Variable<A> unary_operation(const Variable<A>& var, const Op& op);

    template<typename A, typename Op>
    friend Variable<A> binary_operation(const Variable<A>& lhs, const Variable<A>& rhs, const Op& op);


    ///////////////////////////////////////////////////////////////////////////
    ///                          UNARY OPERATIONS                           ///
    ///////////////////////////////////////////////////////////////////////////

    template<typename A>
    friend Variable<A> operator-(const Variable<A>& var);

    Variable<T> exp() const {
        return unary_operation(*this, OperatorRegistry::Exp{});
    }

    Variable<T> log() const {
        return unary_operation(*this, OperatorRegistry::Log{});
    }

    Variable<T> sin() const {
        return unary_operation(*this, OperatorRegistry::Sin{});
    }

    Variable<T> cos() const {
        return unary_operation(*this, OperatorRegistry::Cos{});
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

private:
    std::shared_ptr<VariableImpl<T>> _variable;
};


template<typename T, typename Op>
Variable<T> binary_operation(const Variable<T>& lhs, const Variable<T>& rhs, const Op& op) {
    bool requires_grad = lhs.requires_grad() || rhs.requires_grad();
    Variable<T> out(op(lhs.value(), rhs.value()), requires_grad, false);
    // Variables created by operations are non-leaf

    if (out.requires_grad()) {
        // Note that `lhs_var && rhs_var` should always evaluate to `true` and
        // weak pointers are only chosen to not unnecessarily increase the
        // reference count of the parents of this->_variable (makes debugging easier).
        // This is due to `lhs_wp` or `rhs_wp` being invalid only if `this->_variable`
        // has deleted its parents (note: just deleting parents does not force
        // that lhs._variable nor rhs._variable are invalid), but this is only
        // done in `this->_variable->backward()` after the gradients of all
        // children have been accumulated and `retain_graph=false`. But in this
        // case `_backward_fn` is also set to `nullptr`, thus `_backward_fn`
        //  can no longer be called in `this->_variable->backward()`, therefore
        // `lhs_var && rhs_var` always evaluate to `true` when
        // `this->_variable->backward()` is called.
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
        lhs._variable->add_child(out._variable);
        rhs._variable->add_child(out._variable);
    }

    return out;
}

template<typename T, typename Op>
Variable<T> unary_operation(const Variable<T>& var, const Op& op) {
    Variable<T> out(op(var.value()), var.requires_grad(), false);
    // Variables created by operations are non-leaf

    if (out.requires_grad()) {
        // for discussion of weak pointer usage see `binary_operation`
        out._variable->set_backward_fn([var_wp = std::weak_ptr<VariableImpl<T>>(var._variable), &op](const T& prev_grad) {
            auto var = var_wp.lock();
            if (var)
                return op.backward(*var, prev_grad);
            return std::vector<T>{};
        });

        out._variable->add_parent(var._variable);
        var._variable->add_child(out._variable);
    }

    return out;
}



template<typename T>
Variable<T> operator-(const Variable<T>& var) {
    return unary_operation(var, OperatorRegistry::Neg{});
}



template <typename T>
Variable<T> operator+(const Variable<T>& lhs, const Variable<T>& rhs) {
    return binary_operation(lhs, rhs, OperatorRegistry::Add{});
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
    return binary_operation(lhs, rhs, OperatorRegistry::Sub{});
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
    return binary_operation(lhs, rhs, OperatorRegistry::Mul{});
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
    return binary_operation(lhs, rhs, OperatorRegistry::Div{});
}

template<typename T>
Variable<T> operator/(const Variable<T>& lhs, const T& rhs) {
    return lhs / Variable<T>(rhs, false, false);
}

template<typename T>
Variable<T> operator/(const T& lhs, const Variable<T>& rhs) {
    return Variable<T>(lhs, false, false) / rhs;
}



///////////////////////////////////////////////////////////////////////////
///                              PRINTING                               ///
///////////////////////////////////////////////////////////////////////////

enum class VariableFormatMode {
    Normal,
    Debug
};

// custom formatter of Variable to enable `std::println("{}", variable)` and
// for debug mode `std::println("{:d}", variable)`
template<typename T>
struct std::formatter<Variable<T>> : std::formatter<std::string> {
    VariableFormatMode mode = VariableFormatMode::Normal;

    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin();
        auto end = ctx.end();
        
        // Check for custom formatting option 'd' for debug mode
        if (it != end && *it == 'd') {
            mode = VariableFormatMode::Debug;
            ++it;
        }
        return it;
    }

    auto format(const Variable<T>& var, format_context& ctx) const {
        auto out = ctx.out();
        out = std::format_to(out, "Variable({:.12}", var.value());
        if (var.grad())
            out = std::format_to(out, ", grad={:.12}", var.grad().value());

        if (var.requires_grad())
            out = std::format_to(out, ", requires_grad={}", var.requires_grad());

        out = std::format_to(out, ")");

        bool debug_mode = true;
        if (mode == VariableFormatMode::Debug) {
            out = std::format_to(out, "\n └─ [Debug Info]\n");
            out = std::format_to(out, "     └─ _variable: 0x{:x}\n", reinterpret_cast<uintptr_t>(var.variable().get()));
            out = std::format_to(out, "     └─ _variable.use_count(): {}\n", var.variable().use_count());

            if (var.requires_grad()) {
                out = std::format_to(out, "     └─ Parents: [");
                for (const auto& parent : var.parents()) {
                    out = std::format_to(out, "{:.12} (0x{:x} | {}), ", parent->value(), reinterpret_cast<uintptr_t>(parent.get()), parent.use_count());
                }
                out = std::format_to(out, "]");
            }
        }

        return out;
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Variable<T>& var) {
    os << std::format("{}", var);
    return os;
}
