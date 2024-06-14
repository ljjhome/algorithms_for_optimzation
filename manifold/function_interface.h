#ifndef FUNCTION_INTERFACE_H
#define FUNCTION_INTERFACE_H

#include <Eigen/Dense>
#include "state.h"

template<typename Scalar, typename State, int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class FunctionInterface {
public:
    using StateType = State;
    static constexpr int GradientDim = State::TotalDim;
    static constexpr int EqConstraintsDim = EqConstraints;
    static constexpr int IneqConstraintsDim = IneqConstraints;

    using GradientType = Eigen::Matrix<Scalar, GradientDim, 1>;
    using HessianType = Eigen::Matrix<Scalar, GradientDim, GradientDim>;
    using EqConstraintType = Eigen::Matrix<Scalar, EqConstraints, 1>;
    using IneqConstraintType = Eigen::Matrix<Scalar, IneqConstraints, 1>;
    using EqConstraintGradientType = Eigen::Matrix<Scalar, EqConstraints, GradientDim>;
    using IneqConstraintGradientType = Eigen::Matrix<Scalar, IneqConstraints, GradientDim>;

    virtual Scalar evaluate(const StateType& state) const = 0;
    virtual GradientType gradient(const StateType& state) const = 0;
    virtual HessianType hessian(const StateType& state) const = 0;

    virtual EqConstraintType equalityConstraints(const StateType& state) const = 0;
    virtual IneqConstraintType inequalityConstraints(const StateType& state) const = 0;
    virtual EqConstraintGradientType equalityConstraintsGradient(const StateType& state) const = 0;
    virtual IneqConstraintGradientType inequalityConstraintsGradient(const StateType& state) const = 0;

    virtual ~FunctionInterface() = default;
};

#endif // FUNCTION_INTERFACE_H
