#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>

#include "VariableImpl.hpp"



template<typename T> class Variable;

namespace OperatorRegistry {

    ///////////////////////////////////////////////////////////////////////////
    ///                          BINARY OPERATIONS                          ///
    ///////////////////////////////////////////////////////////////////////////

    struct Add {
        template<typename T>
        T operator()(const T lhs, const T rhs) const { return lhs + rhs; }


        template<typename T>
        std::vector<Variable<T>> backward(const Variable<T>& lhs, const Variable<T>& rhs, const Variable<T>& prev_grad) const {
            return {prev_grad, prev_grad};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& lhs_impl, const std::shared_ptr<VariableImpl<T>>& rhs_impl, const Variable<T>& prev_grad) const {
        //     return {prev_grad, prev_grad};
        // }
    };

    struct Sub {
        template<typename T>
        T operator()(const T lhs, const T rhs) const { return lhs - rhs; }

        template<typename T>
        std::vector<Variable<T>> backward(const Variable<T>& lhs, const Variable<T>& rhs, const Variable<T>& prev_grad) const {
            return {prev_grad, -prev_grad};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& lhs_impl, const std::shared_ptr<VariableImpl<T>>& rhs_impl, const Variable<T>& prev_grad) const {
        //     return {prev_grad, -prev_grad};
        // }
    };

    struct Mul {
        template<typename T>
        T operator()(const T lhs, const T rhs) const { return lhs * rhs; }

        template<typename T>
        std::vector<Variable<T>> backward(const Variable<T>& lhs, const Variable<T>& rhs, const Variable<T>& prev_grad) const {
            return {prev_grad * rhs, prev_grad * lhs};
        }
        
        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& lhs_impl, const std::shared_ptr<VariableImpl<T>>& rhs_impl, const Variable<T>& prev_grad) const {
        //     Variable<T> lhs(lhs_impl);
        //     Variable<T> rhs(rhs_impl);
        //     return {prev_grad * rhs, prev_grad * lhs};
        // }
    };

    struct Div {
        template<typename T>
        T operator()(const T lhs, const T rhs) const { return lhs / rhs; }

        template<typename T>
        std::vector<Variable<T>> backward(const Variable<T>& lhs, const Variable<T>& rhs, const Variable<T>& prev_grad) const {
            return {prev_grad / rhs, prev_grad * -lhs / (rhs * rhs)};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& lhs_impl, const std::shared_ptr<VariableImpl<T>>& rhs_impl, const Variable<T>& prev_grad) const {
        //     Variable<T> lhs(lhs_impl);
        //     Variable<T> rhs(rhs_impl);
        //     return {prev_grad / rhs, prev_grad * -lhs / (rhs * rhs)};;
        // }
    };

    ///////////////////////////////////////////////////////////////////////////
    ///                          UNARY OPERATIONS                           ///
    ///////////////////////////////////////////////////////////////////////////

    struct Neg {
        template<typename T>
        T operator()(const T val) const { return -val; }

        template<typename T>
         std::vector<Variable<T>> backward(const Variable<T>& var, const Variable<T>& prev_grad) const {
            return {-prev_grad};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& var_impl, const Variable<T>& prev_grad) const {
        //     return {-prev_grad};
        // }        
    };

    struct Reciprocal {
        template<typename T>
        T operator()(const T val) const { return 1/val; }

        template<typename T>
         std::vector<Variable<T>> backward(const Variable<T>& var, const Variable<T>& prev_grad) const {
            return {prev_grad * static_cast<T>(-1) / (var * var)};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& var_impl, const Variable<T>& prev_grad) const {
        //     Variable<T> var(var_impl);
        //     return {prev_grad * static_cast<T>(-1) / (var * var)};
        // } 
    };

    struct Abs {
        template<typename T>
        T operator()(const T val) const { return std::abs(val); }

        template<typename T>
         std::vector<Variable<T>> backward(const Variable<T>& var, const Variable<T>& prev_grad) const {
            T sign = var.value() > 0 ? 1 : var.value() < 0 ? -1 : 0;
            return {prev_grad * sign};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& var_impl, const Variable<T>& prev_grad) const {
        //     // Variable<T> var(var_impl);
        //     T sign = var_impl->value() > 0 ? 1 : var_impl->value() < 0 ? -1 : 0;
        //     return {prev_grad * sign};
        // } 
    };

    struct Exp {
        template<typename T>
        T operator()(const T val) const { return std::exp(val); }

        template<typename T>
         std::vector<Variable<T>> backward(const Variable<T>& var, const Variable<T>& prev_grad) const {
            return {prev_grad * var.exp()};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& var_impl, const Variable<T>& prev_grad) const {
        //     Variable<T> var(var_impl);
        //     return {prev_grad * var.exp()};
        // } 
    };

    struct Log {
        template<typename T>
        T operator()(const T val) const { return std::log(val); }

        template<typename T>
         std::vector<Variable<T>> backward(const Variable<T>& var, const Variable<T>& prev_grad) const {
            return {prev_grad * (static_cast<T>(1) / var)};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& var_impl, const Variable<T>& prev_grad) const {
        //     Variable<T> var(var_impl);
        //     return {prev_grad / var};
        // } 
    };

    struct Sin {
        template<typename T>
        T operator()(const T val) const { return std::sin(val); }

        template<typename T>
         std::vector<Variable<T>> backward(const Variable<T>& var, const Variable<T>& prev_grad) const {
            return {prev_grad * var.cos()};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& var_impl, const Variable<T>& prev_grad) const {
        //     Variable<T> var(var_impl);
        //     return {prev_grad * var.cos()};
        // } 
    };

    struct Cos {
        template<typename T>
        T operator()(const T val) const { return std::cos(val); }

        template<typename T>
         std::vector<Variable<T>> backward(const Variable<T>& var, const Variable<T>& prev_grad) const {
            return {prev_grad * -var.sin()};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& var_impl, const Variable<T>& prev_grad) const {
        //     Variable<T> var(var_impl);
        //     return {prev_grad * -var.sin()};
        // } 
    };

    struct Tan {
        template<typename T>
        T operator()(const T val) const { return std::tan(val); }

        template<typename T>
         std::vector<Variable<T>> backward(const Variable<T>& var, const Variable<T>& prev_grad) const {
            return {prev_grad * static_cast<T>(1)/(var.cos() * var.cos())};
        }

        // template<typename T>
        // std::vector<Variable<T>> backward(const std::shared_ptr<VariableImpl<T>>& var_impl, const Variable<T>& prev_grad) const {
        //     Variable<T> var(var_impl);
        //     return {prev_grad / (var.cos() * var.cos())};
        // } 
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

    Variable(const std::shared_ptr<VariableImpl<T>>& variable) : _variable(variable) {}
    
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
    std::optional<Variable<T>> grad() const { return _variable->grad(); }   
    void zero_grad() { _variable->zero_grad(); }
    bool requires_grad() const { return _variable->requires_grad(); }
    bool set_requires_grad(bool v = true) { return _variable->set_requires_grad(v); }
    bool is_leaf() const { return _variable->is_leaf(); }
    const std::shared_ptr<VariableImpl<T>>& variable() const { return _variable; }

    void backward(T prev_grad = 1, bool retain_graph = false, bool create_graph = false) {
        if (create_graph) {
            // When computing higher order derivatives they might depend on 
            // lower order derivatives, i.e. they depend on the computational
            // graph of previous backward calls, thus those previous
            // graphs need to be retained.
            assert(retain_graph && "create_graph required retain_graph");
        }
        _variable->backward(Variable(prev_grad, create_graph, false), retain_graph, nullptr, _variable);
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

    Variable<T> reciprocal() const {
        return unary_operation(*this, OperatorRegistry::Reciprocal{});
    }

    Variable<T> abs() const {
        return unary_operation(*this, OperatorRegistry::Abs{});
    }

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

    Variable<T> tan() const {
        return unary_operation(*this, OperatorRegistry::Tan{});
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
        // out._variable->set_backward_fn([lhs_wp = std::weak_ptr<VariableImpl<T>>(lhs._variable),
        //                                 rhs_wp = std::weak_ptr<VariableImpl<T>>(rhs._variable), &op](const Variable<T>& prev_grad) {
        //     auto lhs_var = lhs_wp.lock();
        //     auto rhs_var = rhs_wp.lock();
        //     if (lhs_var && rhs_var) {
        //         return op.backward(lhs_var, rhs_var, prev_grad);
        //     }
        //     return std::vector<Variable<T>>{};
        // });

        // The lambda function captures the Variables as values, i.e. creates
        // copies of them (and thus increases the use count of the underlying
        // shared pointer), since if they are captured by reference, then those
        // can cause some weird errors at runtime. Not sure why right now. 
        out._variable->set_backward_fn([lhs, rhs, &op](const Variable<T>& prev_grad) {
            return op.backward(lhs, rhs, prev_grad);
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
        // out._variable->set_backward_fn([var_wp = std::weak_ptr<VariableImpl<T>>(var._variable), &op](const Variable<T>& prev_grad) {
        //     auto var = var_wp.lock();
        //     if (var)
        //         return op.backward(var, prev_grad);
        //     return std::vector<Variable<T>>{};
        // });
        out._variable->set_backward_fn([var, &op](const Variable<T>& prev_grad) {
            return op.backward(var, prev_grad);
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

// Custom formatter of Variable to enable `std::println("{}", variable)` and
// for debug mode `std::println("{:d.5}", variable)` with precision
template<typename T>
struct std::formatter<Variable<T>> : std::formatter<std::string> {
    VariableFormatMode _mode = VariableFormatMode::Normal;
    std::optional<int> _precision = std::nullopt;

    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin();
        auto end = ctx.end();
        
        // Check for custom formatting option 'd' for debug mode
        if (it != end && *it == 'd') {
            _mode = VariableFormatMode::Debug;
            ++it;
        }

        // Check for optional precision specification (e.g., ".4")
        if (it != end && *it == '.') {
            ++it;
            int parsed_precision = 0;
            while (it != end && std::isdigit(*it)) {
                parsed_precision = parsed_precision * 10 + (*it - '0');
                ++it;
            }
            _precision = parsed_precision;
        }

        return it;
    }

    auto format(const Variable<T>& var, format_context& ctx) const {
        auto out = ctx.out();

        int precision = _precision.value_or(4);

        if constexpr (std::is_floating_point_v<T>) {
            out = std::format_to(out, "Variable({:.{}g}", var.value(), precision);
            if (var.grad().has_value())
                out = std::format_to(out, ", grad={:.{}g}", var.grad().value().value(), precision);
        } else {
            out = std::format_to(out, "Variable({})", var.value());
            if (var.grad().has_value())
                out = std::format_to(out, ", grad={}", var.grad().value().value());
        }

        if (var.requires_grad())
            out = std::format_to(out, ", requires_grad={}", var.requires_grad());

        out = std::format_to(out, ")");

        if (_mode == VariableFormatMode::Debug) {
            out = std::format_to(out, "\n └─ [Debug Info]\n");
            out = std::format_to(out, "     └─ _variable: 0x{:x}\n", reinterpret_cast<uintptr_t>(var.variable().get()));
            out = std::format_to(out, "     └─ _variable.use_count(): {}\n", var.variable().use_count());
            out = std::format_to(out, "     └─ _variable.has_backward_fn: {}\n", var.variable()->has_backward_fn());

            if (var.requires_grad()) {
                out = std::format_to(out, "     └─ Parents: [");
                for (const auto& parent : var.parents()) {
                    if constexpr (std::is_floating_point_v<T>) {
                        out = std::format_to(out, "{:.{}g} (0x{:x} | {}), ", parent->value(), precision, reinterpret_cast<uintptr_t>(parent.get()), parent.use_count());
                    } else {
                        out = std::format_to(out, "{} (0x{:x} | {}), ", parent->value(), reinterpret_cast<uintptr_t>(parent.get()), parent.use_count());
                    }
                }
                out = std::format_to(out, "]\n");

                out = std::format_to(out, "     └─ Children: [");
                for (const auto& child_wp : var.children()) {
                    if (std::shared_ptr<VariableImpl<T>> child = child_wp.lock()) {
                        if constexpr (std::is_floating_point_v<T>) {
                            out = std::format_to(out, "{:.{}g} (0x{:x} | {}), ", child->value(), precision, reinterpret_cast<uintptr_t>(child.get()), child.use_count() - 1);
                        } else {
                            out = std::format_to(out, "{} (0x{:x} | {}), ", child->value(), reinterpret_cast<uintptr_t>(child.get()), child.use_count() - 1);
                        }
                    }
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
