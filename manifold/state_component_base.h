#ifndef STATE_COMPONENT_BASE_H
#define STATE_COMPONENT_BASE_H

#include <Eigen/Dense>

template<typename Derived, typename Scalar, int Dim>
class StateComponentBase {
public:
    virtual ~StateComponentBase() = default;

    // Pure virtual functions to be implemented by derived classes
    virtual Derived boxPlus(const Eigen::Matrix<Scalar, Dim, 1>& delta) const = 0;
    virtual Eigen::Matrix<Scalar, Dim, 1> boxMinus(const StateComponentBase<Derived, Scalar, Dim>& other) const = 0;
};

#endif // STATE_COMPONENT_BASE_H
