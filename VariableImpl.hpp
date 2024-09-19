#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <cassert>
#include <optional>


template<typename T>
class VariableImpl : public std::enable_shared_from_this<VariableImpl<T>> {
public:
    // VariableImpl() : _value(T(0)), _grad(), _requires_grad(false), _is_leaf(false) {}

    VariableImpl(T value, bool requires_grad = false, bool is_leaf = false)
        : _value(value), _grad(), _requires_grad(requires_grad), _is_leaf(is_leaf) {}

    T value() const { return _value; }
    std::optional<T> grad() const { return _grad; }
    bool requires_grad() const { return _requires_grad; }
    bool set_requires_grad(bool v = true) {
        _is_leaf = true;
        return _requires_grad = v; 
    }
    bool is_leaf() const { return _is_leaf; }

    int num_children() const { 
        int c = 0;
        for (const auto& child_wp : _children) {
            if (!child_wp.expired())
                ++c;
        }
        return c;
     }

    void set_grad(T grad) { _grad = grad; }
    void zero_grad() { _grad = T(0); }
    void add_grad(T grad) { _grad = _grad ? _grad.value() + grad : grad; }

    const std::vector<std::shared_ptr<VariableImpl<T>>>& parents() const { return _parents; }
    const std::vector<std::weak_ptr<VariableImpl<T>>>& children() const { return _children; }

    void add_parent(std::shared_ptr<VariableImpl<T>> parent) {
        if (_requires_grad) {
            _parents.emplace_back(parent);
            parent->add_child(this->shared_from_this());
        }
    }

    void add_child(std::shared_ptr<VariableImpl<T>> child) {
        if (_requires_grad) {
            _children.emplace_back(child);
        }
    }

    void backward(T prev_grad, bool retain_graph) {
        if (!requires_grad())
            return;

        ++_num_bwd_calls;
        // Accumulate all incoming gradients
        add_grad(prev_grad);

        
        int n_children = num_children();

        bool last_bwd_call;
        if (retain_graph) {
            last_bwd_call = (_num_bwd_calls == n_children || n_children == 0);
        } else {
            last_bwd_call = (n_children == 1 || n_children == 0);
        }
        // if the incoming gradients of all children have been accumulated
        // calculate the gradients of the inputs using registered backward functions
        if (last_bwd_call && _backward_fn) {
            std::vector<T> in_grads = _backward_fn(_grad.value());
            size_t n_inputs = _parents.size();
            assert(n_inputs == in_grads.size());

            for (size_t i = 0; i < n_inputs; ++i) {
                _parents[i]->backward(in_grads[i], retain_graph);
            }

            if (!retain_graph) {
                _parents.clear();
                _backward_fn = nullptr; // Clear the lambda to release captured variables
            }
            _num_bwd_calls = 0;

            // only leaf nodes keep their gradients
            if (!is_leaf())
                _grad.reset();
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
    int _num_bwd_calls = 0;
    std::vector<std::shared_ptr<VariableImpl<T>>> _parents;
    std::vector<std::weak_ptr<VariableImpl<T>>> _children;
    std::function<std::vector<T>(const T&)> _backward_fn;
};

