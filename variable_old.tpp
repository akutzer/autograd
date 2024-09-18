// #include "variable.hpp"





// Implement the operators outside the class
#define IMPL_BINARY_OPERATOR(op) \
/* Variable op Variable */ \
template<class A, class B> \
Variable<typename std::common_type<A, B>::type> operator op (const Variable<A>& lhs, const Variable<B>& rhs) { \
    using X = typename std::common_type<A, B>::type; \
    Variable<X> var(lhs.value() op rhs.value()); \
    var.add_parent(lhs); \
    var.add_parent(rhs); \
    return var; \
} \
\
/* Variable op Generic */ \
template<class A, class B> \
Variable<typename std::common_type<A, B>::type> operator op (const Variable<A>& lhs, const B& rhs) { \
    using X = typename std::common_type<A, B>::type; \
    Variable<X> var(lhs._value op rhs); \
    var.parents.push_back(lhs); \
    return var; \
} \
\
/* Generic op Variable */ \
template<class A, class B> \
Variable<typename std::common_type<A, B>::type> operator op (const A& lhs, const Variable<B>& rhs) { \
    using X = typename std::common_type<A, B>::type; \
    Variable<X> var(lhs op rhs._value); \
    var.parents.push_back(rhs); \
    return var; \
}


IMPL_BINARY_OPERATOR(+)
IMPL_BINARY_OPERATOR(-)
IMPL_BINARY_OPERATOR(*)
IMPL_BINARY_OPERATOR(/)

#undef IMPL_BINARY_OPERATOR