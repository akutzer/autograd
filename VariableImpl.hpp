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
    bool set_requires_grad(const bool v = true) {
        _is_leaf = true;
        return _requires_grad = v; 
    }
    bool is_leaf() const { return _is_leaf; }

    // int num_children() const {
    //     // Returns the number of all children which are ancestor of any "living" VariableImpl
    //     int c = 0;
    //     for (const auto& child_wp : _children) {
    //         if (!child_wp.expired())
    //             ++c;
    //     }
    //     return c;
    //  }

    bool is_child(const std::shared_ptr<VariableImpl<T>>& child) const {
        for (const auto& child_wp : _children) {
            auto child_other = child_wp.lock();
            if (child_other && child_other.get() == child.get()) {
                return true;
            }
        }
        return false;
    }        

    void set_grad(const T& grad) { _grad = grad; }
    void zero_grad() { _grad = T(0); }
    void add_grad(const T& grad) { _grad = _grad ? _grad.value() + grad : grad; }

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

    // The backward() function computes and propagates gradients for this variable 
    // in a computational graph. It first accumulates the incoming gradients 
    // from its children, then determines if this is the last backward call 
    // for this variable based on the number of backward calls made and the 
    // number of children it has. If it is the last call and a backward function 
    // is defined, it computes the gradients for its parents using this function 
    // and recursively calls their backward() methods to propagate the gradients 
    // up the graph. If retain_graph is true, the computational graph is preserved 
    // for future backward calls; otherwise, it clears the parent pointers and 
    // the backward function to manage memory efficiently. Only leaf nodes retain 
    // their gradients, while non-leaf nodes reset their gradients to avoid 
    // incorrect accumulation in future calls.


    // The `backward()` function first accumulates the incoming gradients of all children
    // followed by computing the gradients w.r.t. the inputs using `_backward_fn`
    // and propagating these to its parents.
    // The traversal of the computational graph can be though of as a recursive
    // depth-search up until an ancestor `VariableImpl` with multiple children (a branching point).
    // If this ancestor `VariableImpl` has unvisited children the traversal backtracks to
    // a `VariableImpl` with other unvisited parents and continues from there.
    // If all children of the ancestor `VariableImpl` have been visited then this ancestor `VariableImpl`
    // can begin with computing the gradients w.r.t. its inputs and continue
    // the traversal on its first input.
    // If `retain_graph=true`, then the visits to `this` are counted using 
    // `_num_bwd_calls`. If `retain_graph=false` then the children are 
    // deleted after they have propagated to their parents (and they backtrack)

    // Example of the traversal
    //        X
    //        |
    //        A
    //       / \
    //      B   C
    //       \ /
    //        D
    // 
    // Traversal order:
    //   D.backward() -> B.backward() -> A.backward() -> backtrack to B
    //    -> B resets ptr to A -> backtrack to D -> D resets ptr to B
    //    -> B gets deleted since D._parents was the last living shared_pointer to B
    //    -> C.backward() -> A.backward() -> X.backward() -> backtrack to A
    //    -> A resets ptr to X -> C resets ptr to A -> D resets ptr to C

    // The `backward()` function computes and propagates gradients for this variable 
    // in a computational graph. It first accumulates the incoming gradients 
    // from its children, then determines if this is the last backward call 
    // for this variable based on the number of backward calls made and the 
    // number of children it has. If it is the last call and a `_backward_fn` 
    // is defined, it computes the gradients for its parents using `_backward_fn`
    // and recursively calls their `backward()` methods to propagate the gradients 
    // up the graph. If `retain_graph` is `true`, the computational graph is preserved 
    // for future backward calls; otherwise, it clears the parent pointers and 
    // the backward function to manage memory efficiently. Only leaf nodes retain 
    // their gradients, while non-leaf nodes reset their gradients to avoid 
    // incorrect accumulation in future calls.


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
    //         \ /
    //          D
    //
    // Traversal Steps for Backpropagation without retaining the graph:
    // 1. Start from the node D:
    //    - Call D.backward() to begin the backward process.
    //
    // 2. D accumulates its gradient and checks its children:
    //    - Calls B.backward() (its first parent) to propagate gradients upwards.
    //
    // 3. B accumulates the incoming gradient and checks its children:
    //    - Calls A.backward() (its parent) to continue the upward propagation.
    //
    // 4. A accumulates the incoming gradient and checks its children:
    //    - Since its children B & C are still both alive return (i.e. backtrack to B)
    //
    // 5. Child of B returns its backward() call
    //    - B clears the pointer to A
    //    - Since A was the only child B will now clear its parents & backward_fn
    //    - B returns its backward() call (i.e. backtracks to D)
    //
    // 6. The first child of D returns its backward() call 
    //    - D clears the pointer to B
    //    - B gets deleted because D had the last shared pointer to B
    //    - D calls C.backward()
    //
    // 7. C accumulates its gradient and checks its children:
    //    - Calls A.backward() (its parent) to propagate gradients upwards.
    //
    // 8. A accumulates its gradient and checks its children:
    //    - Since C is the only child alive continue the upward propagation
    //
    // 9. X accumulates its gradient and checks its children:
    //    - It has no children so it returns (i.e. backtracks to A)
    //
    //10. Child of A returns its backward() call
    //    - A clears the pointer to X
    //    - Since X was the only child, A will now clear its parents & backward_fn
    //    - A returns its backward() call (i.e. backtracks to C)
    //
    //11. Child of C returns its backward() call
    //    - C clears the pointer to A
    //    - Since A was the only child, C will now clear its parents & backward_fn
    //    - C returns its backward() call (i.e. backtracks to D)
    //
    //12. Child of D returns its backward() call
    //    - D clears the pointer to C
    //    - Since C was the last child, D will now clear its parents & backward_fn
    //    - D returns its backward(), which finishes the Backpropagation

    void backward(const T& prev_grad, bool retain_graph, const std::shared_ptr<VariableImpl<T>>& child = nullptr) {
        if (!requires_grad())
            return;

        // Accumulate all incoming gradients
        add_grad(prev_grad);

        // if (is_child(child))
            // ++_num_bwd_calls;

        ++_num_bwd_calls;
        int n_children = _children.size();
        bool is_last_bwd_call = _num_bwd_calls == n_children || n_children == 0;
        
        // if the incoming gradients of all children have been accumulated
        // calculate the gradients of the inputs using registered backward functions
        if (is_last_bwd_call && _backward_fn) {
            std::vector<T> in_grads = _backward_fn(_grad.value());
            size_t n_inputs = _parents.size();
            assert(n_inputs == in_grads.size());

            for (size_t i = 0; i < n_inputs; ++i) {
                _parents[i]->backward(in_grads[i], retain_graph, this->shared_from_this());
                if (!retain_graph)
                    _parents[i].reset();
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
    std::function<std::vector<T>(const T&)> _backward_fn;
};

