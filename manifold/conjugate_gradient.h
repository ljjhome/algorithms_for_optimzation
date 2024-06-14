#ifndef CONJUGATE_GRADIENT_DESCENT_H
#define CONJUGATE_GRADIENT_DESCENT_H

#include "optimization_method_interface.h"

template<typename Scalar, typename State>
class ConjugateGradientDescent : public OptimizationMethodInterface<Scalar, State> {
public:
    using StateType = typename OptimizationMethodInterface<Scalar, State>::StateType;
    using FunctionType = typename OptimizationMethodInterface<Scalar, State>::FunctionType;
    using GradientType = typename OptimizationMethodInterface<Scalar, State>::GradientType;

    ConjugateGradientDescent()
        : initialized(false), previous_gradient(Eigen::Matrix<Scalar, State::TotalDim, 1>::Zero()),
          previous_direction(Eigen::Matrix<Scalar, State::TotalDim, 1>::Zero()) {}

    virtual GradientType getUpdateDirection(const StateType& current_state, const FunctionType& function) const override {
        GradientType current_gradient = function.gradient(current_state);

        if (!initialized) {
            initialized = true;
            previous_gradient = current_gradient;
            previous_direction = -current_gradient;  // Initial direction is simply the negative gradient
            return previous_direction;
        }

        // Compute Beta using one of the formulae, here using the Polak-Ribiere formula
        Scalar beta = ((current_gradient.transpose() * (current_gradient - previous_gradient)) /
                      (previous_gradient.transpose() * previous_gradient)).value();
        beta = std::max(Scalar(0), beta);  // Ensure beta is non-negative to maintain descent property

        // Update direction
        GradientType direction = -current_gradient + beta * previous_direction;

        // Update previous values for next iteration
        previous_gradient = current_gradient;
        previous_direction = direction;

        return direction;
    }

    mutable bool initialized;
    mutable GradientType previous_gradient;
    mutable GradientType previous_direction;
};

#endif // CONJUGATE_GRADIENT_DESCENT_H
