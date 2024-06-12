#ifndef CONSTRAINED_OPTIMIZER_H
#define CONSTRAINED_OPTIMIZER_H

#include "optimizer_interface.h"
#include "unconstrained_optimizer.h"
#include "constrained_optimization_method_interface.h"
#include "function_interface.h"
#include "concrete_function.h"
#include <iostream>
#include "glog/logging.h"
template<typename Scalar, int Rows, int Cols, int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class ConstrainedOptimizer : public OptimizerInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints> {
public:
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    using FunctionType = FunctionInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>;

    ConstrainedOptimizer(
        UnconstrainedOptimizer<Scalar, Rows, Cols>& unconstrained_optimizer,
        ConstrainedOptimizationMethodInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>& method)
        : unconstrained_optimizer_(unconstrained_optimizer),
          method_(method),
          success_(false), reason_("Not started") {}

    void optimize(MatrixType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        method_.initialize(x, function, config);
        method_.setTolerance(config.common.epsilon_a);

        int current_iteration = 0;
        LOG(INFO) <<"start x: "<<x(0,0)<<","<<x(1,0)<<std::endl;
        LOG(INFO) <<"config.augmented_lagrangian.k_max: "<<config.augmented_lagrangian.k_max<<std::endl;
        for (; current_iteration < config.augmented_lagrangian.k_max; ++current_iteration) {
            LOG(INFO) << "=========current iter :=============== "<< current_iteration<<std::endl;
            auto augmented_function = method_.getAugmentedFunction(function);
            auto augmented_gradient = method_.getAugmentedGradient(function);

            ConcreteFunction<Scalar, Rows, Cols> augmented_function_wrapper(
                augmented_function,
                augmented_gradient,
                [&](const MatrixType& x) -> typename FunctionType::HessianType {
                    return FunctionType::HessianType::Zero(); // Placeholder, as Hessian computation might be complex
                });

            unconstrained_optimizer_.optimize(x, augmented_function_wrapper, config);

            if (unconstrained_optimizer_.isSuccess()) {
                if (method_.hasConverged(x, function)) {
                    success_ = true;
                    LOG(INFO) << "Converged with constraints satisfied"<<std::endl;
                    reason_ = "Converged with constraints satisfied";
                    break;
                }
            } else {
                LOG(INFO) << "Unconstrained optimization failed"<<std::endl;
                reason_ = "Unconstrained optimization failed.";
                // break;
            }
            LOG(INFO) << "x: "<<x(0,0)<<","<<x(1,0)<<std::endl;
            method_.updateParameters(x, function, config);
        }

        if (!success_) {
            reason_ = "Reached maximum iterations without convergence.";
        }

        optimized_variables_ = x;
    }

    bool isSuccess() const override {
        return success_;
    }

    std::string getReason() const override {
        return reason_;
    }

    MatrixType getOptimizedVariables() const override {
        return optimized_variables_;
    }

private:
    UnconstrainedOptimizer<Scalar, Rows, Cols>& unconstrained_optimizer_;
    ConstrainedOptimizationMethodInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>& method_;
    bool success_;
    std::string reason_;
    MatrixType optimized_variables_;
};

#endif // CONSTRAINED_OPTIMIZER_H
