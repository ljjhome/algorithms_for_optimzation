#ifndef CONSTRAINED_OPTIMIZER_H
#define CONSTRAINED_OPTIMIZER_H

#include "optimizer_interface.h"
#include "unconstrained_optimizer.h"
#include "constrained_optimization_method_interface.h"
#include "function_interface.h"
#include "concrete_function.h"
#include <iostream>
#include <glog/logging.h>

template<typename Scalar, typename State,int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class ConstrainedOptimizer : public OptimizerInterface<Scalar, State,EqConstraints,IneqConstraints> {
public:
    using StateType = State;
    using FunctionType = FunctionInterface<Scalar, State,EqConstraints,IneqConstraints>;
    using HessianType = typename FunctionType::HessianType;

    ConstrainedOptimizer(
        UnconstrainedOptimizer<Scalar, State>& unconstrained_optimizer,
        ConstrainedOptimizationMethodInterface<Scalar, State,EqConstraints,IneqConstraints>& method)
        : unconstrained_optimizer_(unconstrained_optimizer),
          method_(method),
          success_(false), reason_("Not started") {}

    void optimize(StateType& state, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        method_.initialize(state, function, config);
        method_.setTolerance(config.common.epsilon_a);

        int current_iteration = 0;
        LOG(INFO) << "start state.";

        for (; current_iteration < config.augmented_lagrangian.k_max; ++current_iteration) {
            LOG(INFO) << "Current iteration: " << current_iteration;
            auto augmented_function = method_.getAugmentedFunction(function);
            auto augmented_gradient = method_.getAugmentedGradient(function);

            ConcreteFunction<Scalar, State> augmented_function_wrapper(
                augmented_function,
                augmented_gradient
                // [&](const StateType& x) -> HessianType {
                //     return HessianType::Zero(); // Placeholder for Hessian computation
                // });
            );

            unconstrained_optimizer_.optimize(state, augmented_function_wrapper, config);

            if (unconstrained_optimizer_.isSuccess()) {
                if (method_.hasConverged(state, function)) {
                    success_ = true;
                    LOG(INFO) << "Converged with constraints satisfied.";
                    reason_ = "Converged with constraints satisfied";
                    break;
                }
            } else {
                LOG(INFO) << "Unconstrained optimization failed.";
                reason_ = "Unconstrained optimization failed.";
            }
            LOG(INFO) << "State update.";
            method_.updateParameters(state, function, config);
        }

        if (!success_) {
            reason_ = "Reached maximum iterations without convergence.";
        }

        optimized_variables_ = state;
    }

    bool isSuccess() const override {
        return success_;
    }

    std::string getReason() const override {
        return reason_;
    }

    StateType getOptimizedVariables() const override {
        return optimized_variables_;
    }

private:
    UnconstrainedOptimizer<Scalar, State>& unconstrained_optimizer_;
    ConstrainedOptimizationMethodInterface<Scalar, State, EqConstraints,IneqConstraints>& method_;
    bool success_;
    std::string reason_;
    StateType optimized_variables_;
};

#endif // CONSTRAINED_OPTIMIZER_H
