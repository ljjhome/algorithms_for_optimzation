#ifndef ANGULAR_VELOCITY_CONTROL_INPUT_H
#define ANGULAR_VELOCITY_CONTROL_INPUT_H

#include "control_input_base.h"

template<typename Scalar, int Dim = 3>
class AngularVelocityControlInput : public ControlInputBase<Scalar, Dim> {
public:
    using VectorType = Eigen::Matrix<Scalar, Dim, 1>;

    AngularVelocityControlInput(const VectorType& angular_velocity)
        : angular_velocity_(angular_velocity) {}

    Eigen::Matrix<Scalar, Dim, 1> getValue() const override {
        return angular_velocity_;
    }

private:
    VectorType angular_velocity_;
};

#endif // ANGULAR_VELOCITY_CONTROL_INPUT_H
