#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <cassert>
#include <optional>


template<class T>
class VariableImpl {
public:
    VariableImpl() : _value(T(0)), _grad(T(0)), _requires_grad(false), _is_leaf(false) {}

    VariableImpl(T value, bool requires_grad = false, bool is_leaf = false)
        : _value(value), _grad(T(0)), _requires_grad(requires_grad), _is_leaf(is_leaf) {}

    T value() const { return _value; }
    std::optional<T> grad() const { return _grad; }
    bool requires_grad() const { return _requires_grad; }
    bool is_leaf() const { return _is_leaf; }

    void set_grad(T grad) { _grad = grad; }
    void zero_grad() { _grad = T(0); }
    void add_grad(T grad) { _grad = _grad ? _grad.value() + grad : grad; }

    const std::vector<std::weak_ptr<VariableImpl<T>>>& parents() const { return _parents; }

    void add_parent(std::shared_ptr<VariableImpl> parent) {
        if (_requires_grad) {
            _parents.emplace_back(parent);
        }
    }

    void backward(T prev_grad, bool retain_graph) {
        if (!requires_grad())
            return;

        // Accumulate incoming gradient
        if (is_leaf())
            add_grad(prev_grad);

        // Calculate gradients of inputs using registered backward functions
        if (_backward_fn) {
            std::vector<T> out_grads = _backward_fn(prev_grad);
            size_t n_inputs = _parents.size();
            assert(n_inputs == out_grads.size());

            for (size_t i = 0; i < n_inputs; ++i) {
                if (auto parent = _parents[i].lock())
                    parent->backward(out_grads[i], retain_graph);
            }

            if (!retain_graph)
                _parents.clear();
        }
    }

    void set_backward_fn(std::function<std::vector<T>(const T&)> backward_fn) {
        _backward_fn = std::move(backward_fn);
    }

private:
    T _value;
    std::optional<T> _grad;
    bool _requires_grad;
    bool _is_leaf; // only leaf Variables will have their grad populated during a call to backward()
    std::vector<std::weak_ptr<VariableImpl<T>>> _parents;  // weak_ptr to prevent circular references
    std::function<std::vector<T>(const T&)> _backward_fn;
};

