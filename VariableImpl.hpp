#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <cassert>


template<class T>
class VariableImpl {
public:
    VariableImpl() : _value(T(0)), _grad(T(0)), _requires_grad(false), _is_leaf(false), _backward_fn(std::function<std::vector<T>(const T&)>()) {}

    VariableImpl(T value, bool requires_grad = false, bool is_leaf = false, std::function<std::vector<T>(const T&)> backward_fn = std::function<std::vector<T>(const T&)>())
        : _value(value), _grad(T(0)), _requires_grad(requires_grad), _is_leaf(is_leaf), _backward_fn(backward_fn) {}
    

    T value() const { return _value; }
    T grad() const { return _grad; }
    bool requires_grad() const { return _requires_grad; }
    bool is_leaf() const { return _is_leaf; }

    void set_grad(T grad) { _grad = grad; }
    void zero_grad() { set_grad(T(0)); }
    void add_grad(T grad) { _grad += grad; }

    const std::vector<std::shared_ptr<VariableImpl<T>>>& parents() const {
        return _parents;
    }

    void add_parent(std::shared_ptr<VariableImpl> parent) {
        if (_requires_grad)
            _parents.push_back(parent);
    }   

    void backward(T prev_grad, bool retain_graph) {
        if (!requires_grad()) {
            return;
        }

        // accumulate incoming gradient
        if (is_leaf())
            _grad += prev_grad;

        // calculate gradients of inputs
        if (_backward_fn) {
            std::vector<T> out_grads = _backward_fn(prev_grad);

            size_t n_inputs = _parents.size();
            assert(n_inputs == out_grads.size());

            for (int i = 0; i < n_inputs; ++i) {
                auto& parent = _parents.at(i);
                parent->backward(out_grads.at(i), retain_graph);
            }
            
            if (!retain_graph)
                _parents.clear();
        }
    }

private:
    T _value;
    T _grad;
    bool _requires_grad;
    bool _is_leaf; // only leaf Variables will have their grad populated during a call to backward()
    std::vector<std::shared_ptr<VariableImpl<T>>> _parents;
    std::function<std::vector<T>(const T&)> _backward_fn;
};

