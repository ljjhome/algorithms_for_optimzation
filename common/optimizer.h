// optimizer.h
#include "common/function_interface.h"
#include "common/optimization_method_interface.h"
#include "common/line_search_interface.h"
#include <glog/logging.h>

template<typename Scalar, int Rows, int Cols>
class Optimizer {
public:
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;

    Optimizer(const FunctionInterface<Scalar, Rows, Cols>& function,
              const OptimizationMethodInterface<Scalar, Rows, Cols>& method,
              const LineSearchInterface<Scalar, Rows, Cols>& lineSearch)
        : function_(function), method_(method), lineSearch_(lineSearch), success_(false), reason_("Not started") {}

    void optimize(MatrixType& x, int max_iterations, Scalar epsilon_a, Scalar epsilon_r, Scalar epsilon_g) {
        int current_iteration = 0;
        Scalar f_x_k = function_.evaluate(x);
        Scalar f_x_k_1 = f_x_k;
        MatrixType d_f_x, x_k_1;

        for (; current_iteration < max_iterations; ++current_iteration) {
            d_f_x = function_.gradient(x);
            MatrixType update_direction = method_.getUpdateDirection(x, function_);
            Scalar alpha = lineSearch_.search(x, update_direction, function_, 50);

            x_k_1 = x + alpha * update_direction;
            f_x_k_1 = function_.evaluate(x_k_1);

            // Termination conditions
            if (f_x_k - f_x_k_1 < epsilon_a) {
                success_ = true;
                reason_ = "Converged with absolute improvement";
                break;
            }
            if (f_x_k - f_x_k_1 < epsilon_r * std::abs(f_x_k)) {
                success_ = true;
                reason_ = "Converged with relative improvement";
                break;
            }
            if (d_f_x.norm() < epsilon_g) {
                success_ = true;
                reason_ = "Converged with gradient magnitude";
                break;
            }

            x = x_k_1;
            f_x_k = f_x_k_1;
            LOG(INFO) << "Iteration " << current_iteration << ": " << x.transpose();
        }

        if (current_iteration >= max_iterations) {
            success_ = false;
            reason_ = "Reached maximum iterations without convergence.";
        }

        optimized_variables_ = x;
    }

    bool isSuccess() const {
        return success_;
    }

    std::string getReason() const {
        return reason_;
    }

    MatrixType getOptimizedVariables() const {
        return optimized_variables_;
    }

private:
    const FunctionInterface<Scalar, Rows, Cols>& function_;
    const OptimizationMethodInterface<Scalar, Rows, Cols>& method_;
    const LineSearchInterface<Scalar, Rows, Cols>& lineSearch_;
    bool success_;
    std::string reason_;
    MatrixType optimized_variables_;
};