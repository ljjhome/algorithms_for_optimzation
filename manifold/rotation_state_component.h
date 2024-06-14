#ifndef ROTATION_STATE_COMPONENT_H
#define ROTATION_STATE_COMPONENT_H

#include "state_component_base.h"

template<typename Scalar>
class RotationStateComponent : public StateComponentBase<RotationStateComponent<Scalar>, Scalar, 3> {
public:
    using RotationType = Eigen::Matrix<Scalar, 3, 3>;

    RotationStateComponent(const RotationType& rotation) : rotation_(rotation) {}

    RotationStateComponent() : rotation_(Eigen::Matrix<Scalar, 3, 3>::Identity()) {}



    RotationStateComponent<Scalar> boxPlus(const Eigen::Matrix<Scalar, 3, 1>& delta) const override {
        Eigen::AngleAxis<Scalar> delta_rot(delta.norm(), delta.normalized());
        RotationType new_rotation = rotation_ * delta_rot.toRotationMatrix();
        return RotationStateComponent<Scalar>(new_rotation);
    }

    Eigen::Matrix<Scalar, 3, 1> boxMinus(const StateComponentBase<RotationStateComponent<Scalar>, Scalar, 3>& other) const override {
        const auto& otherRot = static_cast<const RotationStateComponent<Scalar>&>(other);
        Eigen::Matrix<Scalar, 3, 3> delta_rot = otherRot.rotation_.transpose() * rotation_;
        Eigen::AngleAxis<Scalar> angle_axis(delta_rot);
        return angle_axis.angle() * angle_axis.axis();
    }

    RotationType getRotation() const {
        return rotation_;
    }

    void setRotation(const RotationType& rotation) {
        rotation_ = rotation;
    }

    static constexpr int getDimension() {
        return 3;
    }

private:
    RotationType rotation_;
};

#endif // ROTATION_STATE_COMPONENT_H
