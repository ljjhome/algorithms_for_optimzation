#include <iostream>
#include <Eigen/Dense>
#include "manifold/state.h"
#include "manifold/vector_state_component.h"
#include "manifold/rotation_state_component.h"

using Scalar = double;

int main() {
    using Vector3d = Eigen::Matrix<Scalar, 3, 1>;
    using Rotation3d = Eigen::Matrix<Scalar, 3, 3>;

    // Define state components
    Vector3d v1 = Vector3d::Random();
    Vector3d delta_v = Vector3d::Random() * 0.1;
    Rotation3d r1 = Rotation3d::Identity();
    Eigen::Matrix<Scalar, 3, 1> delta_r;
    delta_r << 0.1, 0.2, 0.3;

    // Create vector and rotation state components
    VectorStateComponent<Scalar, 3> vec1(v1);
    RotationStateComponent<Scalar> rot1(r1);

    // Create state
    State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>> state(vec1, rot1);

    // Perform boxPlus operation
    Eigen::Matrix<Scalar, State<Scalar, VectorStateComponent<Scalar, 3>, RotationStateComponent<Scalar>>::TotalDim, 1> delta;
    delta << delta_v, delta_r;
    auto new_state = state.boxPlus(delta);

    // Extract components from new state
    auto new_vec1 = std::get<0>(new_state.getComponents()).getVector();
    auto new_rot1 = std::get<1>(new_state.getComponents()).getRotation();

    std::cout << "Original Vector: \n" << v1 << std::endl;
    std::cout << "Delta Vector: \n" << delta_v << std::endl;
    std::cout << "New Vector: \n" << new_vec1 << std::endl;

    std::cout << "Original Rotation: \n" << r1 << std::endl;
    std::cout << "Delta Rotation: \n" << delta_r.transpose() << std::endl;
    std::cout << "New Rotation: \n" << new_rot1 << std::endl;

    // Perform boxMinus operation
    auto delta_computed = new_state.boxMinus(state);
    std::cout << "Computed Delta: \n" << delta_computed << std::endl;

    return 0;
}
