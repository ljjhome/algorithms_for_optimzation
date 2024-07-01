#ifndef UNCONSTRAINED_OPTIMIZER_H
#define UNCONSTRAINED_OPTIMIZER_H

#include "optimizer_interface.h"
#include "line_search_interface.h"
#include "optimization_method_interface.h"
#include "common/parameters/unified_optimizer_config.h"
#include <iostream>
#include <glog/logging.h>

template<typename Scalar, typename State>
class UnconstrainedOptimizer : public OptimizerInterface<Scalar, State> {
public:
    using StateType = typename OptimizerInterface<Scalar, State>::StateType;
    using FunctionType = typename OptimizerInterface<Scalar, State>::FunctionType;
    using GradientType = typename FunctionType::GradientType;

    UnconstrainedOptimizer(
        OptimizationMethodInterface<Scalar, State>& method,
        const LineSearchInterface<Scalar, State>& line_search)
        : method_(method), line_search_(line_search), success_(false), reason_("Not started") {}

    void optimize(StateType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        int current_iteration = 0;
        function.unconstrainedUpdate();
        Scalar f_x_k = function.evaluate(x);
        Scalar f_x_k_1 = f_x_k;
        GradientType d_f_x;
        StateType x_k_1 = x;

        for (; current_iteration < config.common.max_iterations; ++current_iteration) {
            // LOG(INFO) << "========unconstrained iter: ==================" << current_iteration;
            d_f_x = function.gradient(x);
            GradientType update_direction = method_.getUpdateDirection(x, function);
            // update_direction = update_direction.normalized();
            // LOG(INFO) << "dir : " << update_direction.transpose() << std::endl;
            // Scalar alpha = line_search_.search(x, update_direction, function, 1000);
            // LOG(INFO) << "alpha: " << alpha << std::endl;
            Scalar alpha = 1;
            auto update = Eigen::Matrix<Scalar, State::TotalDim, 1>(alpha * update_direction);
            x_k_1 = x.boxPlus(update);
            f_x_k_1 = function.evaluate(x_k_1);
            // LOG(INFO) << "f1: " <<f_x_k<<", f2: " << f_x_k_1 << std::endl;

            // LOG(INFO) << "Iteration " << current_iteration << ": " << x_k_1 << std::endl;
            // Termination conditions
            if (std::abs(f_x_k - f_x_k_1) < config.common.epsilon_a) {
                success_ = true;
                LOG(INFO) << "Converged with absolute improvement: " << current_iteration<<", alpha: "<<alpha;
                reason_ = "Converged with absolute improvement";
                break;
            }
            if (d_f_x.norm() < config.common.epsilon_g) {
                success_ = true;
                LOG(INFO) << "Converged with gradient magnitude" << std::endl;
                reason_ = "Converged with gradient magnitude";
                break;
            }
            x = x_k_1;
            f_x_k = f_x_k_1;
            function.unconstrainedUpdate();
        }

        if (!success_) {
            LOG(INFO) << "unconstrained reach max iterations" << std::endl;
            reason_ = "Reached maximum iterations without convergence.";
        }
        // method_.ResetParameters();
        optimized_variables_ = x;
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
    OptimizationMethodInterface<Scalar, State>& method_;
    const LineSearchInterface<Scalar, State>& line_search_;
    bool success_;
    std::string reason_;
    StateType optimized_variables_;
};

#endif // UNCONSTRAINED_OPTIMIZER_H
