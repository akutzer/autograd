#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <cassert>
#include <optional>


template<typename T> class Variable;

template<typename T>
class VariableImpl : public std::enable_shared_from_this<VariableImpl<T>> {
public:
    // VariableImpl() : _value(T(0)), _grad(), _requires_grad(false), _is_leaf(false) {}

    VariableImpl(T value, bool requires_grad = false, bool is_leaf = false)
        : _value(value), _grad(), _requires_grad(requires_grad), _is_leaf(is_leaf) {}

    // VariableImpl(T value, bool requires_grad = false, bool is_leaf = false)
    //     : _value(value), _grad(), _requires_grad(requires_grad), _is_leaf(is_leaf) {}

    T value() const { return _value; }
    std::optional<Variable<T>> grad() const { return _grad; }
    bool requires_grad() const { return _requires_grad; }
    bool set_requires_grad(const bool requires_grad = true) {
        _is_leaf = true;
        if (!requires_grad) {
            _parents.clear();
            _children.clear();
            _backward_fn = nullptr;
        }
        return _requires_grad = requires_grad;
    }
    bool is_leaf() const { return _is_leaf; }


    bool is_child(const std::shared_ptr<VariableImpl<T>>& child) const {
        for (const auto& child_wp : _children) {
            auto child_other = child_wp.lock();
            if (child_other && child_other.get() == child.get()) {
                return true;
            }
        }
        return false;
    }        

    // void set_grad(const T& grad) { _grad = grad; }
    // void set_grad(const std::shared_ptr<VariableImpl<T>>& grad) { _grad = grad; }
    void set_grad(const Variable<T>& grad) { _grad = grad; }
    void zero_grad() { _grad = Variable<T>(0, false, false); }
    // void add_grad(const T& grad) { _grad = _grad.has_value() ? _grad.value() + grad : grad; }
    // void add_grad(const std::shared_ptr<VariableImpl<T>>& grad) { 
    //     if (_grad.value())
    //         _grad.value()->_value = _grad.value()->_value + grad->value();
    //     else
    //         _grad = grad;
    // }
    void add_grad(const Variable<T>& grad) { _grad = _grad.has_value() ? _grad.value() + grad : grad; }
    

    const std::vector<std::shared_ptr<VariableImpl<T>>>& parents() const { return _parents; }
    const std::vector<std::weak_ptr<VariableImpl<T>>>& children() const { return _children; }

    void add_parent(const std::shared_ptr<VariableImpl<T>>& parent) {
        if (_requires_grad) {
            _parents.emplace_back(parent);
        }
    }

    void add_child(const std::shared_ptr<VariableImpl<T>>& child) {
        if (_requires_grad) {
            _children.emplace_back(child);
        }
    }

    bool is_part_of_graph(const std::shared_ptr<VariableImpl<T>>& root) const {
        if (this == root.get())
            return true;

        for (const auto& child_wp : _children) {
            if (auto child = child_wp.lock(); child && child->is_part_of_graph(root))
                return true;
        }
        return false;
    }

    int num_children_in_graph(const std::shared_ptr<VariableImpl<T>>& root) const {
        int num = 0;
        for (const auto& child_wp : _children) {
            if (auto child = child_wp.lock(); child && child->is_part_of_graph(root))
                ++num;
        }
        return num;
    }


    // The `backward()` function computes and propagates gradients for this variable 
    // in a computational graph. It first accumulates the incoming gradients 
    // from its children, then determines if this is the last backward call 
    // for this variable based on the number of backward calls made and the 
    // number of children it has. If it is the last call and a `_backward_fn` 
    // is defined, it computes the gradients for its parents using `_backward_fn`
    // and recursively calls their `backward()` methods to propagate the gradients 
    // up the graph. If `retain_graph` is `true`, the computational graph is preserved 
    // for future backward calls; otherwise, it clears the parent pointers and 
    // the backward function to release memory resources. Only leaf nodes retain 
    // their gradients, while non-leaf nodes reset their gradients to avoid 
    // incorrect accumulation in future calls.
    //
    //
    ///////////////////////////////////////////////////////////////////////////
    ///                Example for the backward() traversal                 ///
    ///////////////////////////////////////////////////////////////////////////
    //
    // Graph Structure:
    //
    //          X
    //          |
    //          A
    //         / \ 
    //        B   C
    //       / \ /
    //      E   D <- Variable(D)
    //
    // Traversal Steps for Backpropagation without retaining the graph:
    //  1. Start from the Variable(D).backward():
    //     - Call D.backward() with `root=true to begin the backward process.
    //
    //  2. In D:
    //     - D checks how many of its children are part of the computational graph -> [].
    //     - Since D is the root of the computational graph it accumulates the 
    //       incoming gradient.
    //     - Since D is the root it computes the gradients w.r.t. the inputs.
    //     - Calls B.backward() (its first parent) to propagate its gradient upwards.
    //
    //  3. In B:
    //     - B checks how many of its children are part of the computational graph -> [D].
    //        (E is not an ancestor of the root)
    //     - Since D is a valid child it accumulates the incoming gradient.
    //     - Since B has now accumulated all incoming gradients it computes the gradients w.r.t. the inputs.
    //     - Calls A.backward() to propagate its gradient upwards.
    //
    //  4. In A:
    //     - A checks how many of its children are part of the computational graph -> [B,C].
    //     - Since B is a valid child it accumulates the incoming gradient.
    //     - Since the gradient of C is still missing it returns the backward() call to D
    //
    //  5. In D:
    //     - Deletes the reference to B (If nothing else references B, then it gets deleted)
    //     - Calls C.backward() (its second parent) to propagate its gradient upwards.
    //
    //  6. In C:
    //     - C checks how many of its children are part of the computational graph -> [D].
    //     - Since D is a valid child it accumulates the incoming gradient.
    //     - Since C has now accumulated all incoming gradients it computes the gradients w.r.t. the inputs.
    //     - Calls A.backward() to propagate its gradient upwards.
    //
    //  7. In A:
    //     - Since C is a valid child it accumulates the incoming gradient.
    //     - Since A has now accumulated all incoming gradients it computes the gradients w.r.t. the inputs.
    //     - Calls X.backward() to propagate its gradient upwards.
    //
    //  8. In X:
    //     - X checks how many of its children are part of the computational graph -> [A].
    //     - Since X is a valid child it accumulates the incoming gradient.
    //     - Since X has no `_backward_fn` it returns the backward() call to A.
    //
    //  9. In A:
    //     - Deletes the reference to X and returns the backward() call to C.
    //
    // 10. In C:
    //     - Deletes the reference to A and returns the backward() call to D.
    //
    // 11. In D:
    //     - Deletes the reference to C and returns the backward() call to Variable(D)
    //       finishing the backward process.
    //
    void backward(const Variable<T>& prev_grad, bool retain_graph, const std::shared_ptr<VariableImpl<T>>& child = nullptr, const std::shared_ptr<VariableImpl<T>>& root = nullptr) {
        // If a variable has no parents that require gradients, we do not need
        // to propagate gradients at all
        if (!requires_grad())
            return;
        
        // On the first backwards call traverse the computational graph downwards
        // towards the root and check how many of the children that are alive are
        // an ancestor of the root. We do this only on the first call because
        // if the graph is not retained, then during the accumulation process
        // of the incoming gradients some children might already get deleted
        // thus reducing the number of children that are part of the graph
        if (_num_bwd_calls == -1) {
            _children_in_graph = num_children_in_graph(root);
            _num_bwd_calls = 0;
        }

        // Accumulate all incoming gradients from valid children or if `this` is
        // the root of the computational graph
        bool is_root = this == root.get();
        if (is_root || is_child(child)) {
            add_grad(prev_grad);
            ++_num_bwd_calls;
        }

        // if the incoming gradients of all children have been accumulated
        // calculate the gradients of the inputs using registered backward functions
        bool is_last_bwd_call = is_root || _num_bwd_calls == _children_in_graph;
        if (is_last_bwd_call && _backward_fn) {
            // If one incoming `prev_grad` has `requires_grad = true`, then all
            // outgoing gradients will also have `requires_grad = true`, thus
            // they will create a new computational graph 
            bool create_graph = _grad.has_value() && _grad.value().requires_grad();
            std::vector<Variable<T>> in_grads = _backward_fn(_grad.value());
            
            size_t n_inputs = _parents.size();
            assert(n_inputs == in_grads.size());

            for (size_t i = 0; i < n_inputs; ++i) {
                auto& in_grad = in_grads[i];
                if (!create_graph) {
                    in_grad.set_requires_grad(false);
                }
                _parents[i]->backward(in_grads[i], retain_graph, this->shared_from_this(), root);
                if (!retain_graph)
                    _parents[i].reset();
            }

            if (!retain_graph) {
                _parents.clear();
                // _children.clear();
                _backward_fn = nullptr;
            }
            _num_bwd_calls = -1;
            _children_in_graph = 0;

            // only leaf nodes keep their gradients
            if (!is_leaf())
                _grad.reset();
        }
    }

    // void set_backward_fn(std::function<Variable<T>(const Variable<T>&)> backward_fn) {
    //     _backward_fn = std::move(backward_fn);
    // }

    void set_backward_fn(std::function<std::vector<Variable<T>>(const Variable<T>&)> backward_fn) {
        _backward_fn = std::move(backward_fn);
    }

private:
    T _value;
    // std::optional<T> _grad;
    std::optional<Variable<T>> _grad;
    bool _requires_grad;
    bool _is_leaf; // only leaf Variables will have their grad populated during a call to backward()
    int _num_bwd_calls = -1;
    int _children_in_graph = 0;
    // VariableImpl stores its parents as a shared pointer, enforcing their
    // presence for the `_backward_fn`, while keeping their children only as
    // weak pointers, since if the children are part of the computation
    // graph (i.e. an ancestor) of the final scalar variable on which `backward()` is called,
    // then they are kept alive by their children (i.e. the grandchildren of `this`),
    // and so on, which are ultimately kept alive by the VariableImpl on which 
    // the initial `backward()` was called. However, if not and they go out of scope,
    // then those children might get deleted, but this is no problem, since then
    // they are not part of the computation graph
    std::vector<std::shared_ptr<VariableImpl<T>>> _parents;
    std::vector<std::weak_ptr<VariableImpl<T>>> _children;
    std::function<std::vector<Variable<T>>(const Variable<T>&)> _backward_fn;
};
