#include <iostream>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>
#include "common/concrete_function.h"
#include "common/unconstrained_optimizer.h"
#include "common/backtracking_line_search.h"
#include "common/gradient_descent.h"
#include "common/adam.h"
#include "common/penalty_method.h"
#include "common/augmented_lagrangian_method.h"
#include "common/constrained_optimizer.h"
#include "common/parameters/unified_optimizer_config.h"
#include <glog/logging.h>

// Objective function: f(x) = x1^2 + x2^2
double objectiveFunction(const Eigen::Matrix<double, 2, 1>& x) {
    return x.squaredNorm();
}

// Gradient of the objective function: grad_f(x) = [2*x1, 2*x2]
Eigen::Matrix<double, 2, 1> objectiveGradient(const Eigen::Matrix<double, 2, 1>& x) {
    return 2 * x;
}

// Equality constraint: h(x) = x1 + x2 - 1
Eigen::Matrix<double, 1, 1> equalityConstraint(const Eigen::Matrix<double, 2, 1>& x) {
    Eigen::Matrix<double, 1, 1> h;
    h << x.sum() - 1;
    return h;
}

// Gradient of the equality constraint: grad_h(x) = [1, 1]
Eigen::Matrix<double, 1, 2> equalityConstraintGradient(const Eigen::Matrix<double, 2, 1>& x) {
    Eigen::Matrix<double, 1, 2> grad_h;
    grad_h << 1, 1;
    return grad_h;
}

// Inequality constraint: g(x) = [x1, x2] (x1 >= 0, x2 >= 0)
Eigen::Matrix<double, 2, 1> inequalityConstraint(const Eigen::Matrix<double, 2, 1>& x) {
    return -x;
}

// Gradient of the inequality constraint: grad_g(x) = [[1, 0], [0, 1]]
Eigen::Matrix<double, 2, 2> inequalityConstraintGradient(const Eigen::Matrix<double, 2, 1>& x) {
    return -Eigen::Matrix<double, 2, 2>::Identity();
}


int main(int argc, char** argv) {
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

    using Scalar = double;
    const int Rows = 2;
    const int Cols = 1;
    const int EqConstraints = 1;
    const int IneqConstraints = 2;

    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    using EqConstraintsType = Eigen::Matrix<Scalar, EqConstraints, 1>;
    using IneqConstraintsType = Eigen::Matrix<Scalar, IneqConstraints, 1>;
    using EqConstraintsGradientType = Eigen::Matrix<Scalar, EqConstraints, Rows>;
    using IneqConstraintsGradientType = Eigen::Matrix<Scalar, IneqConstraints, Rows>;

    auto func = [](const MatrixType& x) -> Scalar {
        return objectiveFunction(x);
    };

    auto grad = [](const MatrixType& x) -> MatrixType {
        return objectiveGradient(x);
    };

    auto eq_constraints = [](const MatrixType& x) -> EqConstraintsType {
        return equalityConstraint(x);
    };

    auto eq_constraints_grad = [](const MatrixType& x) -> EqConstraintsGradientType {
        return equalityConstraintGradient(x);
    };

    auto ineq_constraints = [](const MatrixType& x) -> IneqConstraintsType {
        return inequalityConstraint(x);
    };

    auto ineq_constraints_grad = [](const MatrixType& x) -> IneqConstraintsGradientType {
        return inequalityConstraintGradient(x);
    };

    auto hess = [](const MatrixType& x) -> Eigen::Matrix<Scalar, Rows, Rows> {
        return Eigen::Matrix<Scalar, Rows, Rows>::Zero(); // Placeholder Hessian matrix
    };

    ConcreteFunction<Scalar, Rows, Cols, EqConstraints, IneqConstraints> myFunction(
        func, grad, hess,eq_constraints, ineq_constraints, eq_constraints_grad, ineq_constraints_grad
    );

    // Load configuration from YAML file
    UnifiedOptimizerConfig config;
    config.loadFromYaml("../config/config.yaml");

    // Choose the unconstrained optimizer (Gradient Descent with Backtracking Line Search)
    GradientDescent<Scalar, Rows, Cols> gradient_descent(config.gradient_descent.learning_rate);
    BacktrackingLineSearch<Scalar, Rows, Cols> line_search;
    UnconstrainedOptimizer<Scalar, Rows, Cols> unconstrained_optimizer(gradient_descent, line_search);

    // Choose the constrained optimization method (Penalty Method)
    AugmentedLagrangianMethod<Scalar, Rows, Cols, EqConstraints, IneqConstraints> penalty_method;

    // Create the constrained optimizer
    ConstrainedOptimizer<Scalar, Rows, Cols, EqConstraints, IneqConstraints> constrained_optimizer(
        unconstrained_optimizer, penalty_method
    );

    MatrixType x;
    x <<-5, 12;

    constrained_optimizer.optimize(x, myFunction, config);

    if (constrained_optimizer.isSuccess()) {
        std::cout << "Optimization succeeded: " << constrained_optimizer.getOptimizedVariables().transpose() << std::endl;
    } else {
        std::cout << "Optimization failed: " << constrained_optimizer.getReason() << std::endl;
    }

    return 0;
}
