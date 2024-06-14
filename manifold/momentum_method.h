#ifndef MOMENTUM_METHOD_H
#define MOMENTUM_METHOD_H

#include "optimization_method_interface.h"

template<typename Scalar, typename State>
class MomentumMethod : public OptimizationMethodInterface<Scalar, State> {
public:
    using StateType = typename OptimizationMethodInterface<Scalar, State>::StateType;
    using FunctionType = typename OptimizationMethodInterface<Scalar, State>::FunctionType;
    using GradientType = typename OptimizationMethodInterface<Scalar, State>::GradientType;

    MomentumMethod(Scalar learningRate = 0.1, Scalar momentum = 0.1)
        : learningRate_(learningRate), momentum_(momentum),
          initialized(false), velocity(Eigen::Matrix<Scalar, State::TotalDim, 1>::Zero()) {}

    virtual GradientType getUpdateDirection(const StateType& current_state, const FunctionType& function) const override {
        GradientType current_gradient = function.gradient(current_state);

        if (!initialized) {
            initialized = true;
            velocity = -learningRate_ * current_gradient;  // Initial velocity
            return velocity;
        }

        // Update velocity with the momentum term
        velocity = momentum_ * velocity - learningRate_ * current_gradient;

        return velocity;
    }

private:
    Scalar learningRate_;   // The step size for each iteration
    Scalar momentum_;       // Momentum coefficient to control the influence of the previous updates
    mutable bool initialized;
    mutable GradientType velocity;  // This stores the accumulated gradient direction with momentum
};

#endif // MOMENTUM_METHOD_H
