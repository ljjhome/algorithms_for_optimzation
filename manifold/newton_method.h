#ifndef NEWTON_METHOD_H
#define NEWTON_METHOD_H

#include "optimization_method_interface.h"
#include <Eigen/QR>
#include <glog/logging.h>

template <typename Scalar, typename State, int ResidualDim = Eigen::Dynamic>
class NewtonMethod : public OptimizationMethodInterface<Scalar, State, ResidualDim>
{
public:
    using StateType = typename OptimizationMethodInterface<Scalar, State, ResidualDim>::StateType;
    using FunctionType = FunctionInterface<Scalar, State, ResidualDim>;
    using GradientType = typename OptimizationMethodInterface<Scalar, State, ResidualDim>::GradientType;
    using ResidualType = typename FunctionType::ResidualType;
    using ResidualJacobianType = typename FunctionType::ResidualJacobianType;
    using HessianType = Eigen::Matrix<Scalar, State::TotalDim, State::TotalDim>;

    NewtonMethod() = default;

    virtual GradientType getUpdateDirection(const StateType &current_state, const FunctionType &function) const override
    {
        // Compute the gradient
        GradientType gradient = function.gradient(current_state);

        // Compute the Hessian
        HessianType hessian = function.hessian(current_state);

        // Solve the linear system for the update direction using LDLT decomposition
        Eigen::LDLT<HessianType> ldlt(hessian);
        if (ldlt.info() != Eigen::Success)
        {
            LOG(ERROR) << "LDLT decomposition failed!";
            return GradientType::Zero(); // Return zero direction if decomposition fails
        }
        GradientType update_direction = -ldlt.solve(gradient);
        LOG(INFO) << "Update direction: \n" << update_direction;
        return update_direction;
    }

    void ResetParameters() override
    {
        // Newton's method does not use parameters like Adam, so nothing to reset.
    }
};

#endif // NEWTON_METHOD_H
