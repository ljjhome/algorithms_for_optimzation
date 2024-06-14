#ifndef AUGMENTED_LAGRANGIAN_METHOD_H
#define AUGMENTED_LAGRANGIAN_METHOD_H

#include "constrained_optimization_method_interface.h"
#include <cmath>
#include <glog/logging.h>

template<typename Scalar, typename State,int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class AugmentedLagrangianMethod : public ConstrainedOptimizationMethodInterface<Scalar, State,EqConstraints,IneqConstraints> {
public:
    using StateType = typename ConstrainedOptimizationMethodInterface<Scalar, State>::StateType;
    using FunctionType = typename ConstrainedOptimizationMethodInterface<Scalar, State, EqConstraints,IneqConstraints>::FunctionType;
    using GradientType = typename FunctionType::GradientType;
    using EqConstraintsType = Eigen::Matrix<Scalar, FunctionType::EqConstraintsDim, 1>;
    using IneqConstraintsType = Eigen::Matrix<Scalar, FunctionType::IneqConstraintsDim, 1>;

    void initialize(const StateType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        lambda_eq_ = EqConstraintsType::Zero();
        lambda_ineq_ = IneqConstraintsType::Zero();
        penalty_parameter_ = config.augmented_lagrangian.penalty_parameter_init;
    }

    Scalar getPenalty(const FunctionType& function, const StateType& x) const override {
        Scalar eq_penalty = function.equalityConstraints(x).squaredNorm();
        Scalar ineq_penalty = (function.inequalityConstraints(x).array().max(0).matrix()).squaredNorm();
        return eq_penalty + ineq_penalty;
    }

    void updateParameters(const StateType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        lambda_eq_ += penalty_parameter_ * function.equalityConstraints(x);
        lambda_ineq_ += penalty_parameter_ * function.inequalityConstraints(x).array().max(0).matrix();
        penalty_parameter_ *= config.augmented_lagrangian.gamma;
    }

    bool hasConverged(const StateType& x, const FunctionType& function) const override {
        LOG(INFO) << "check constraints satisfied: " << function.equalityConstraints(x).norm() << ", " << function.inequalityConstraints(x).array().max(0).sum() << std::endl;
        return function.equalityConstraints(x).norm() < tol_ &&
               function.inequalityConstraints(x).array().max(0).sum() < tol_;
    }

    void setTolerance(Scalar tol) override {
        tol_ = tol;
    }

    std::function<Scalar(const StateType&)> getAugmentedFunction(const FunctionType& function) const override {
        return [this, &function](const StateType& x) -> Scalar {
            Scalar f_val = function.evaluate(x);
            Scalar eq_penalty = function.equalityConstraints(x).squaredNorm();
            Scalar ineq_penalty = (function.inequalityConstraints(x).array().max(0).matrix()).squaredNorm();
            Scalar eq_lagrange = lambda_eq_.dot(function.equalityConstraints(x));
            Scalar ineq_lagrange = lambda_ineq_.dot(function.inequalityConstraints(x).array().max(0).matrix());
            return f_val + penalty_parameter_ * (eq_penalty + ineq_penalty) + eq_lagrange + ineq_lagrange;
        };
    }

    std::function<GradientType(const StateType&)> getAugmentedGradient(const FunctionType& function) const override {
        return [this, &function](const StateType& x) -> GradientType {
            GradientType grad = function.gradient(x);
            GradientType eq_penalty_grad = function.equalityConstraintsGradient(x).transpose() * function.equalityConstraints(x);
            GradientType ineq_penalty_grad = function.inequalityConstraintsGradient(x).transpose() * (function.inequalityConstraints(x).array().max(0).matrix());
            GradientType eq_lagrange_grad = function.equalityConstraintsGradient(x).transpose() * lambda_eq_;
            GradientType ineq_lagrange_grad = function.inequalityConstraintsGradient(x).transpose() * lambda_ineq_;
            return grad + 2 * penalty_parameter_ * (eq_penalty_grad + ineq_penalty_grad) + eq_lagrange_grad + ineq_lagrange_grad;
        };
    }

private:
    Scalar penalty_parameter_;
    Scalar tol_ = 1e-6;
    mutable EqConstraintsType lambda_eq_;
    mutable IneqConstraintsType lambda_ineq_;
};

#endif // AUGMENTED_LAGRANGIAN_METHOD_H
