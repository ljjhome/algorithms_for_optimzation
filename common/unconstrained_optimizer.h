#ifndef UNCONSTRAINED_OPTIMIZER_H
#define UNCONSTRAINED_OPTIMIZER_H

#include "common/optimizer_interface.h"
#include "common/line_search_interface.h"
#include "common/optimization_method_interface.h"
#include <iostream>
#include <glog/logging.h>
template<typename Scalar, int Rows, int Cols>
class UnconstrainedOptimizer : public OptimizerInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizerInterface<Scalar, Rows, Cols>::MatrixType;
    using FunctionType = typename OptimizerInterface<Scalar, Rows, Cols>::FunctionType;

    UnconstrainedOptimizer(
        const OptimizationMethodInterface<Scalar, Rows, Cols>& method,
        const LineSearchInterface<Scalar, Rows, Cols>& line_search)
        : method_(method), line_search_(line_search), success_(false), reason_("Not started") {}

    void optimize(MatrixType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) override {
        int current_iteration = 0;
        Scalar f_x_k = function.evaluate(x);
        Scalar f_x_k_1 = f_x_k;
        MatrixType d_f_x, x_k_1;

        for (; current_iteration < config.common.max_iterations; ++current_iteration) {
            LOG(INFO)<<"unconstrained iter: "<< current_iteration;
            d_f_x = function.gradient(x);
            MatrixType update_direction = method_.getUpdateDirection(x, function);
            LOG(INFO) << "dir : "<<update_direction(0,0)<<","<<update_direction(1,0)<<std::endl;
            Scalar alpha = line_search_.search(x, update_direction, function, config.common.max_iterations);
            LOG(INFO) << "alpha: "<<alpha<<std::endl;
            LOG(INFO) << "x_k_1: "<<x_k_1(0,0)<<","<<x_k_1(1,0)<<std::endl;
            auto tt = alpha * update_direction;
            LOG(INFO) << "alpha * update_direction:  "<<tt(0,0)<<","<<tt(1,0)<<std::endl;
            x_k_1 = x + alpha * update_direction;
            f_x_k_1 = function.evaluate(x_k_1);
            LOG(INFO) << "f: "<<f_x_k_1<<std::endl;
            
            
            LOG(INFO) << "Iteration " << current_iteration << ": " << x.transpose() << std::endl;
            // Termination conditions
            if (f_x_k - f_x_k_1 < config.common.epsilon_a) {
                success_ = true;
                LOG(INFO) << "Converged with absolute improvement"<<std::endl;
                reason_ = "Converged with absolute improvement";
                break;
            }
            // if (f_x_k - f_x_k_1 < config.common.epsilon_r * std::abs(f_x_k)) {
            //     success_ = true;
            //     LOG(INFO) << "Converged with relative improvement"<<std::endl;
            //     reason_ = "Converged with relative improvement";
            //     break;
            // }
            if (d_f_x.norm() < config.common.epsilon_g) {
                success_ = true;
                LOG(INFO) << "Converged with gradient magnitude"<<std::endl;
                reason_ = "Converged with gradient magnitude";
                break;
            }
            x = x_k_1;
            f_x_k = f_x_k_1;

            
        }

        if (!success_) {
            LOG(INFO) << "unconstrain reach max iters"<<std::endl;
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
    const OptimizationMethodInterface<Scalar, Rows, Cols>& method_;
    const LineSearchInterface<Scalar, Rows, Cols>& line_search_;
    bool success_;
    std::string reason_;
    MatrixType optimized_variables_;
};

#endif // UNCONSTRAINED_OPTIMIZER_H
