#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include "optimization_method_interface.h"

template<typename Scalar, typename State>
class GradientDescent : public OptimizationMethodInterface<Scalar, State> {
public:
    using StateType = typename OptimizationMethodInterface<Scalar, State>::StateType;
    using FunctionType = typename OptimizationMethodInterface<Scalar, State>::FunctionType;
    using GradientType = typename OptimizationMethodInterface<Scalar, State>::GradientType;

    GradientType getUpdateDirection(const StateType& current_state, const FunctionType& function) const override {
        // Perform a gradient descent update
        return -function.gradient(current_state);
    }
};

#endif // GRADIENT_DESCENT_H
