#include <string>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <functional>
#include <unistd.h>
#include <glog/logging.h>
#include "common/function_interface.h"
#include "common/concrete_function.h"
#include "common/parameters/unified_optimizer_config.h"
#include "common/backtracking_line_search.h"
#include "common/gradient_descent.h"
#include "common/unconstrained_optimizer.h"

double quadfunc(const Eigen::Matrix<double, 2, 1> &x) {
    return pow(x(0, 0), 2) + x(0, 0) * x(1, 0) + pow(x(1, 0), 2);
}

Eigen::Matrix<double, 2, 1> dquadfunc(const Eigen::Matrix<double, 2, 1> &x) {
    Eigen::Matrix<double, 2, 1> res;
    res(0, 0) = 2 * x(0, 0) + x(1, 0);
    res(1, 0) = 2 * x(1, 0) + x(0, 0);
    return -res;
}

int main(int argc, char **argv) {
    std::cout << "hello " << std::endl;

    using Scalar = double;
    const int Rows = 2;
    const int Cols = 1;
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    using HessType = Eigen::Matrix<Scalar, Rows, Rows>;

    // Load configuration
    UnifiedOptimizerConfig config;
    config.loadFromYaml("../config/config.yaml");

    // Define the function, gradient, and Hessian
    auto func = [](const MatrixType& x) -> Scalar {
        return quadfunc(x);
    };

    auto grad = [](const MatrixType& x) -> MatrixType {
        return dquadfunc(x);
    };

    auto hess = [](const MatrixType& x) -> HessType {
        return HessType::Zero();
    };

    // Create the function object
    ConcreteFunction<Scalar, Rows, Cols> myFunction(func, grad, hess);

    // Create the line search method
    BacktrackingLineSearch<Scalar, Rows, Cols> lineSearch;

    // Create the gradient descent method
    GradientDescent<Scalar, Rows, Cols> gradient_descent;

    // Create the optimizer
    UnconstrainedOptimizer<Scalar, Rows, Cols> optimizer(myFunction, gradient_descent, lineSearch);

    // Setup logging
    std::string log_dir = "data/log/";

    if (access(log_dir.c_str(), 0) == -1) {
        std::string command = "mkdir -p " + log_dir;
        system(command.c_str());

        google::InitGoogleLogging(argv[0]);
        google::SetLogDestination(google::INFO, log_dir.c_str());
        google::SetLogDestination(google::ERROR, "");
        google::SetLogDestination(google::WARNING, "");
        google::SetLogDestination(google::FATAL, "");
        google::SetStderrLogging(google::INFO);
    } else {
        google::InitGoogleLogging(argv[0]);
        google::SetLogDestination(google::INFO, log_dir.c_str());
        google::SetLogDestination(google::ERROR, "");
        google::SetLogDestination(google::WARNING, "");
        google::SetLogDestination(google::FATAL, "");
        google::SetStderrLogging(google::INFO);
    }

    MatrixType x;
    x << 1, 2;

    optimizer.optimize(x, config);

    std::cout << "reasen: "<<optimizer.getReason()<<std::endl;
    if (optimizer.isSuccess()) {
        std::cout << "Optimization succeeded: " << optimizer.getOptimizedVariables().transpose() << std::endl;
        
    } else {
        std::cout << "Optimization failed: " << optimizer.getReason() << std::endl;
    }

    return 0;
}
