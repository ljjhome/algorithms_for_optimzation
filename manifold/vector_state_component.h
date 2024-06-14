#ifndef VECTOR_STATE_COMPONENT_H
#define VECTOR_STATE_COMPONENT_H

#include "state_component_base.h"

template<typename Scalar, int Dim>
class VectorStateComponent : public StateComponentBase<VectorStateComponent<Scalar, Dim>, Scalar, Dim> {
public:
    using VectorType = Eigen::Matrix<Scalar, Dim, 1>;

    VectorStateComponent(const VectorType& vector) : vector_(vector) {}

    VectorStateComponent() : vector_(Eigen::Matrix<Scalar, Dim, 1>::Zero()) {}

    VectorStateComponent<Scalar, Dim> boxPlus(const Eigen::Matrix<Scalar, Dim, 1>& delta) const override {
        VectorType new_vector = vector_ + delta;
        return VectorStateComponent<Scalar, Dim>(new_vector);
    }

    Eigen::Matrix<Scalar, Dim, 1> boxMinus(const StateComponentBase<VectorStateComponent<Scalar, Dim>, Scalar, Dim>& other) const override {
        const auto& otherVec = static_cast<const VectorStateComponent<Scalar, Dim>&>(other);
        return vector_ - otherVec.vector_;
    }

    VectorType getVector() const {
        return vector_;
    }

    void setVector(const VectorType& vector) {
        vector_ = vector;
    }

    static constexpr int getDimension() {
        return Dim;
    }

private:
    VectorType vector_;
};

#endif // VECTOR_STATE_COMPONENT_H
