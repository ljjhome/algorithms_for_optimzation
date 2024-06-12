#ifndef PENALTY_METHOD_H
#define PENALTY_METHOD_H

#include "constrained_optimization_method_interface.h"
#include <cmath>
#include <glog/logging.h>
template<typename Scalar, int Rows, int Cols, int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class PenaltyMethod : public ConstrainedOptimizationMethodInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints> {
public:
    using MatrixType = typename ConstrainedOptimizationMethodInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>::MatrixType;
    using FunctionType = typename ConstrainedOptimizationMethodInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>::FunctionType;

    void initialize(const MatrixType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        penalty_parameter_ = config.penalty_method.penalty_parameter_init;
    }

    Scalar getPenalty(const FunctionType& function, const MatrixType& x) const override {
        
        Scalar eq_penalty = function.equalityConstraints(x).squaredNorm();
        Scalar ineq_penalty = (function.inequalityConstraints(x).array().max(0).matrix()).squaredNorm();
        LOG(INFO) << "eq_penalty: "<<eq_penalty<<","<< "ineq penalty: "<< ineq_penalty<<"total: "<<eq_penalty + ineq_penalty<<", param mult: "<<penalty_parameter_*(eq_penalty + ineq_penalty);
        
        return eq_penalty + ineq_penalty;
    }

    void updateParameters(const MatrixType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        penalty_parameter_ *= config.penalty_method.gamma;
        LOG(INFO) << "penalty parameter: "<<penalty_parameter_<<std::endl;
    }

    bool hasConverged(const MatrixType& x, const FunctionType& function) const override {
        LOG(INFO) << "function.equalityConstraints(x).norm(): "<<function.equalityConstraints(x).norm()<<std::endl;
        LOG(INFO) << "function.inequalityConstraints(x).array().max(0).sum(): "<<function.inequalityConstraints(x).array().max(0).sum()<<std::endl;
        return function.equalityConstraints(x).norm() < tol_ &&
               function.inequalityConstraints(x).array().max(0).sum() < tol_;
    }

    void setTolerance(Scalar tol) override {
        tol_ = tol;
    }

    std::function<Scalar(const MatrixType&)> getAugmentedFunction(
        const FunctionType& function) const override {
        return [this, &function](const MatrixType& x) -> Scalar {
            LOG(INFO) << "function.evaluate(x): "<<function.evaluate(x);
            return function.evaluate(x) + penalty_parameter_ * getPenalty(function, x);
        };
    }

    std::function<MatrixType(const MatrixType&)> getAugmentedGradient(
        const FunctionType& function) const override {
        return [this, &function](const MatrixType& x) -> MatrixType {
            MatrixType grad = function.gradient(x);
            MatrixType eq_penalty_grad = function.equalityConstraintsGradient(x).transpose() * function.equalityConstraints(x);
            MatrixType ineq_penalty_grad = function.inequalityConstraintsGradient(x).transpose() * (function.inequalityConstraints(x).array().max(0).matrix());
            LOG(INFO)<<"grad:"<<grad(0,0)<<","<<grad(1,0)<<", eq_grad: "<<eq_penalty_grad(0,0)<<","<<eq_penalty_grad(1,0)<<",ineq_grad:"<<ineq_penalty_grad(0,0)<<","<<ineq_penalty_grad(1,0);
            return (grad + 2 * penalty_parameter_ * (eq_penalty_grad + ineq_penalty_grad));
        };
    }

private:
    Scalar penalty_parameter_;
    Scalar tol_ = 1e-6;
};

#endif // PENALTY_METHOD_H
