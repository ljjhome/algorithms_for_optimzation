#ifndef INTERIOR_POINT_METHOD_H
#define INTERIOR_POINT_METHOD_H

#include "constrained_optimization_method_interface.h"
#include <cmath>

template<typename Scalar, int Rows, int Cols, int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class InteriorPointMethod : public ConstrainedOptimizationMethodInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints> {
public:
    using MatrixType = typename ConstrainedOptimizationMethodInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>::MatrixType;
    using FunctionType = typename ConstrainedOptimizationMethodInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>::FunctionType;

    void initialize(const MatrixType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        barrier_parameter_ = config.interior_point.barrier_parameter_init;
    }

    Scalar getPenalty(const FunctionType& function, const MatrixType& x) const override {
        Scalar ineq_penalty = 0;
        auto ineq_constraints = function.inequalityConstraints(x);
        for (int i = 0; i < ineq_constraints.size(); ++i) {
            if (ineq_constraints[i] > 0) {
                ineq_penalty += -std::log(ineq_constraints[i]);
            }
        }
        return ineq_penalty;
    }

    void updateParameters(const MatrixType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        barrier_parameter_ *= config.interior_point.gamma;
    }

    bool hasConverged(const MatrixType& x, const FunctionType& function) const override {
        return function.equalityConstraints(x).norm() < tol_ &&
               function.inequalityConstraints(x).minCoeff() > 0 &&
               function.inequalityConstraints(x).array().abs().sum() < tol_;
    }

    void setTolerance(Scalar tol) override {
        tol_ = tol;
    }

    std::function<Scalar(const MatrixType&)> getAugmentedFunction(
        const FunctionType& function) const override {
        return [this, &function](const MatrixType& x) -> Scalar {
            return function.evaluate(x) + barrier_parameter_ * getPenalty(function, x);
        };
    }

    std::function<MatrixType(const MatrixType&)> getAugmentedGradient(
        const FunctionType& function) const override {
        return [this, &function](const MatrixType& x) -> MatrixType {
            MatrixType grad = function.gradient(x);
            MatrixType ineq_penalty_grad = function.inequalityConstraintsGradient(x).transpose() * (function.inequalityConstraints(x).array().inverse().matrix());
            return grad - barrier_parameter_ * ineq_penalty_grad;
        };
    }

private:
    Scalar barrier_parameter_;
    Scalar tol_ = 1e-6;
};

#endif // INTERIOR_POINT_METHOD_H
