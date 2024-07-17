#ifndef GAUSS_NEWTON_METHOD_H
#define GAUSS_NEWTON_METHOD_H

#include "optimization_method_interface.h"
#include <Eigen/QR>
#include <glog/logging.h>
template <typename Scalar, typename State, int ResidualDim = Eigen::Dynamic>
class GaussNewtonMethod : public OptimizationMethodInterface<Scalar, State, ResidualDim>
{
public:
    using StateType = typename OptimizationMethodInterface<Scalar, State, ResidualDim>::StateType;
    using FunctionType = FunctionInterface<Scalar, State, ResidualDim>;
    using GradientType = typename OptimizationMethodInterface<Scalar, State, ResidualDim>::GradientType;
    using ResidualType = typename FunctionType::ResidualType;
    using ResidualJacobianType = typename FunctionType::ResidualJacobianType;
    using HessianType = Eigen::Matrix<Scalar, State::TotalDim, State::TotalDim>;

    GaussNewtonMethod() = default;

    virtual GradientType getUpdateDirection(const StateType &current_state, const FunctionType &function) const override
    {

        // Compute residuals and Jacobian
        ResidualType residuals = function.residuals(current_state);
        ResidualJacobianType J = function.residualJacobian(current_state);

        // Compute the Gauss-Newton Hessian approximation with regularization
        Scalar lambda = 0.0; // small regularization parameter
        HessianType H = J.transpose() * J + lambda * HessianType::Identity();
        // LOG(INFO) << "Hessian: \n"
        //             << H;
        // Compute the gradient (J^T * residuals)
        GradientType g = J.transpose() * residuals;
        // auto tt = (J.transpose() * J).inverse() * J.transpose()*residuals;
        

        // Solve the linear system for the update direction using LDLT decomposition
        Eigen::LDLT<HessianType> ldlt(H);
        if (ldlt.info() != Eigen::Success)
        {
            LOG(ERROR) << "LDLT decomposition failed!";
            return GradientType::Zero(); // Return zero direction if decomposition fails
        }
        GradientType update_direction = -ldlt.solve(g);
        // GradientType update_direction = - (J.transpose() * J).inverse() * J.transpose()*residuals;
            LOG(INFO)<<"(J^T J )^-1 J^T \n"<<update_direction;
        return update_direction;
    }

    void ResetParameters() override
    {
        // Gauss-Newton does not use parameters like Adam, so nothing to reset.
    }
};

#endif // GAUSS_NEWTON_METHOD_H
