#include <iostream>
#include <Eigen/Dense>
#include <glog/logging.h>
#include "manifold/state.h"
#include "manifold/vector_state_component.h"
#include "manifold/rotation_state_component.h"
#include "manifold/function_interface.h"
#include "manifold/concrete_function.h"
#include "manifold/penalty_method.h"
#include "manifold/unconstrained_optimizer.h"
#include "manifold/gradient_descent.h"
#include "manifold/backtracking_line_search.h"
#include "common/parameters/unified_optimizer_config.h"

// Define the quadratic function
double quadFunction(const State<double, VectorStateComponent<double, 3>, RotationStateComponent<double>>& state) {
    auto v = std::get<0>(state.getComponents()).getVector();
    auto r = std::get<1>(state.getComponents()).getRotation();
    return v.squaredNorm() + r.squaredNorm();
}

// Define the gradient of the quadratic function
typename State<double, VectorStateComponent<double, 3>, RotationStateComponent<double>>::GradientType quadGradient(const State<double, VectorStateComponent<double, 3>, RotationStateComponent<double>>& state) {
    typename State<double, VectorStateComponent<double, 3>, RotationStateComponent<double>>::GradientType grad;
    grad << 2 * std::get<0>(state.getComponents()).getVector(), 2 * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(std::get<1>(state.getComponents()).getRotation().data());
    return grad;
}

// Define equality constraints
Eigen::Matrix<double, 1, 1> equalityConstraints(const State<double, VectorStateComponent<double, 3>, RotationStateComponent<double>>& state) {
    auto v = std::get<0>(state.getComponents()).getVector();
    Eigen::Matrix<double, 1, 1> eq;
    eq(0, 0) = v.sum() - 1.0; // Example: sum of vector elements should be 1
    return eq;
}

// Define the gradient of the equality constraints
Eigen::Matrix<double, 1, 12> equalityConstraintsGradient(const State<double, VectorStateComponent<double, 3>, RotationStateComponent<double>>& state) {
    Eigen::Matrix<double, 1, 12> eq_grad = Eigen::Matrix<double, 1, 12>::Zero();
    eq_grad.block<1, 3>(0, 0) = Eigen::RowVector3d::Ones();
    return eq_grad;
}

// Define inequality constraints
Eigen::Matrix<double, 1, 1> inequalityConstraints(const State<double, VectorStateComponent<double, 3>, RotationStateComponent<double>>& state) {
    auto v = std::get<0>(state.getComponents()).getVector();
    Eigen::Matrix<double, 1, 1> ineq;
    ineq(0, 0) = 1.0 - v.norm(); // Example: norm of vector elements should be less than or equal to 1
    return ineq;
}

// Define the gradient of the inequality constraints
Eigen::Matrix<double, 1, 12> inequalityConstraintsGradient(const State<double, VectorStateComponent<double, 3>, RotationStateComponent<double>>& state) {
    Eigen::Matrix<double, 1, 12> ineq_grad = Eigen::Matrix<double, 1, 12>::Zero();
    auto v = std::get<0>(state.getComponents()).getVector();
    ineq_grad.block<1, 3>(0, 0) = -v.transpose() / v.norm();
    return ineq_grad;
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    using Scalar = double;
    using Vector3d = Eigen::Matrix<Scalar, 3, 1>;
    using Rotation3d = Eigen::Matrix<Scalar, 3, 3>;

    // Define state components
    Vector3d v1 = Vector3d::Random();
    Rotation3d r1 = Rotation3d::Identity();

    // Create vector and rotation state components
    VectorStateComponent<Scalar, 3> vec1(v1);
    RotationStateComponent<Scalar> rot1(r1);

    // Create state
    State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>> state(vec1, rot1);

    // Define function and gradient
    auto func = [](const State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>& state) { return quadFunction(state); };
    auto grad = [](const State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>& state) { return quadGradient(state); };

    auto eq_constraints = [](const State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>& state) { return equalityConstraints(state); };
    auto ineq_constraints = [](const State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>& state) { return inequalityConstraints(state); };

    auto eq_constraints_grad = [](const State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>& state) { return equalityConstraintsGradient(state); };
    auto ineq_constraints_grad = [](const State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>& state) { return inequalityConstraintsGradient(state); };

    ConcreteFunction<Scalar, State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>> quadFunc(func, grad, nullptr, eq_constraints, ineq_constraints, eq_constraints_grad, ineq_constraints_grad);

    // Define optimization method and line search
    GradientDescent<Scalar, State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>> gradientDescent;
    BacktrackingLineSearch<Scalar, State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>> lineSearch;

    // Define unconstrained optimizer
    UnconstrainedOptimizer<Scalar, State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>> unconstrainedOptimizer(gradientDescent, lineSearch);

    // Define penalty method
    PenaltyMethod<Scalar, State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>> penaltyMethod;

    // Define unified optimizer config
    UnifiedOptimizerConfig config;
    config.common.max_iterations = 100;
    config.common.epsilon_a = 1e-6;
    config.common.epsilon_r = 1e-6;
    config.common.epsilon_g = 1e-6;
    config.penalty_method.penalty_parameter_init = 1.0;
    config.penalty_method.gamma = 2.0;

    // Initialize penalty method
    penaltyMethod.initialize(state, quadFunc, config);

    // Perform constrained optimization
    State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>> current_state = state;

    for (int k = 0; k < config.common.max_iterations; ++k) {
        LOG(INFO) << "Penalty method iteration: " << k;
        auto augmented_function = penaltyMethod.getAugmentedFunction(quadFunc);
        auto augmented_gradient = penaltyMethod.getAugmentedGradient(quadFunc);

        ConcreteFunction<Scalar, State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>> augmentedFunc(
            augmented_function, augmented_gradient, nullptr, eq_constraints, ineq_constraints, eq_constraints_grad, ineq_constraints_grad
        );

        unconstrainedOptimizer.optimize(current_state, augmentedFunc, config);

        if (penaltyMethod.hasConverged(current_state, quadFunc)) {
            LOG(INFO) << "Converged at iteration: " << k;
            break;
        }

        penaltyMethod.updateParameters(current_state, quadFunc, config);
    }

    if (unconstrainedOptimizer.isSuccess()) {
        LOG(INFO) << "Optimization succeeded.";
        auto optimized_variables = unconstrainedOptimizer.getOptimizedVariables();
        LOG(INFO) << "Optimized state: " << optimized_variables;
    } else {
        LOG(INFO) << "Optimization failed: " << unconstrainedOptimizer.getReason();
    }

    return 0;
}
