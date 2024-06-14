#ifndef PENALTY_METHOD_H
#define PENALTY_METHOD_H

#include "constrained_optimization_method_interface.h"
#include <cmath>
#include <glog/logging.h>

template<typename Scalar, typename State>
class PenaltyMethod : public ConstrainedOptimizationMethodInterface<Scalar, State> {
public:
    using StateType = typename ConstrainedOptimizationMethodInterface<Scalar, State>::StateType;
    using FunctionType = typename ConstrainedOptimizationMethodInterface<Scalar, State>::FunctionType;
    using ScalarType = Scalar;

    PenaltyMethod(Scalar initial_penalty = 1.0, Scalar gamma = 1.5)
        : penalty_parameter_(initial_penalty), gamma_(gamma), tol_(1e-6) {}

    void initialize(const StateType& state, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        penalty_parameter_ = config.penalty_method.penalty_parameter_init;
    }

    Scalar getPenalty(const FunctionType& function, const StateType& state) const override {
        Scalar eq_penalty = function.equalityConstraints(state).squaredNorm();
        Scalar ineq_penalty = (function.inequalityConstraints(state).array().max(0).matrix()).squaredNorm();
        LOG(INFO) << "eq_penalty: " << eq_penalty << ", ineq penalty: " << ineq_penalty 
                  << ", total: " << eq_penalty + ineq_penalty << ", param mult: " << penalty_parameter_ * (eq_penalty + ineq_penalty);
        return eq_penalty + ineq_penalty;
    }

    void updateParameters(const StateType& state, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        penalty_parameter_ *= gamma_;
        LOG(INFO) << "Updated penalty parameter: " << penalty_parameter_;
    }

    bool hasConverged(const StateType& state, const FunctionType& function) const override {
        bool isConverged = function.equalityConstraints(state).norm() < tol_ &&
                           function.inequalityConstraints(state).array().max(0).sum() < tol_;
        LOG(INFO) << "Convergence check: " << isConverged;
        return isConverged;
    }

    std::function<Scalar(const StateType&)> getAugmentedFunction(const FunctionType& function) const override {
        return [this, &function](const StateType& state) -> Scalar {
            return function.evaluate(state) + penalty_parameter_ * getPenalty(function, state);
        };
    }

    std::function<typename StateType::GradientType(const StateType&)> getAugmentedGradient(const FunctionType& function) const override {
        return [this, &function](const StateType& state) -> typename StateType::GradientType {
            auto grad = function.gradient(state);
            auto eq_penalty_grad = function.equalityConstraintsGradient(state).transpose() * function.equalityConstraints(state);
            auto ineq_penalty_grad = function.inequalityConstraintsGradient(state).transpose() * (function.inequalityConstraints(state).array().max(0).matrix());
            return grad + 2 * penalty_parameter_ * (eq_penalty_grad + ineq_penalty_grad);
        };
    }

private:
    Scalar penalty_parameter_;
    Scalar gamma_;
    Scalar tol_;
};

#endif // PENALTY_METHOD_H
