#ifndef ACCELERATION_CONTROL_INPUT_H
#define ACCELERATION_CONTROL_INPUT_H

#include "control_input_base.h"

template<typename Scalar, int Dim = 3>
class AccelerationControlInput : public ControlInputBase<Scalar, Dim> {
public:
    using VectorType = Eigen::Matrix<Scalar, Dim, 1>;

    AccelerationControlInput(const VectorType& acceleration)
        : acceleration_(acceleration) {}

    Eigen::Matrix<Scalar, Dim, 1> getValue() const override {
        return acceleration_;
    }

private:
    VectorType acceleration_;
};

#endif // ACCELERATION_CONTROL_INPUT_H
