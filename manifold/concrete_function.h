#ifndef CONCRETE_FUNCTION_H
#define CONCRETE_FUNCTION_H

#include "function_interface.h"
#include <functional>

template <typename Scalar, typename State, int ResidualDim = Eigen::Dynamic, int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class ConcreteFunction : public FunctionInterface<Scalar, State, ResidualDim, EqConstraints, IneqConstraints>
{
public:
    using StateType = State;
    using GradientType = typename FunctionInterface<Scalar, State, ResidualDim, EqConstraints, IneqConstraints>::GradientType;
    using HessianType = typename FunctionInterface<Scalar, State, ResidualDim, EqConstraints, IneqConstraints>::HessianType;
    using EqConstraintType = typename FunctionInterface<Scalar, State, ResidualDim, EqConstraints, IneqConstraints>::EqConstraintType;
    using IneqConstraintType = typename FunctionInterface<Scalar, State, ResidualDim, EqConstraints, IneqConstraints>::IneqConstraintType;
    using EqConstraintGradientType = typename FunctionInterface<Scalar, State, ResidualDim, EqConstraints, IneqConstraints>::EqConstraintGradientType;
    using IneqConstraintGradientType = typename FunctionInterface<Scalar, State, ResidualDim, EqConstraints, IneqConstraints>::IneqConstraintGradientType;
    using ResidualType = typename FunctionInterface<Scalar, State, ResidualDim, EqConstraints, IneqConstraints>::ResidualType;
    using ResidualJacobianType = typename FunctionInterface<Scalar, State, ResidualDim, EqConstraints, IneqConstraints>::ResidualJacobianType;

    ConcreteFunction(
        std::function<Scalar(const StateType &)> func,
        std::function<GradientType(const StateType &)> grad,
        std::function<ResidualType(const StateType &)> residuals = nullptr,
        std::function<ResidualJacobianType(const StateType &)> residual_jacobian = nullptr,
        std::function<bool()> unconstrained_update = nullptr,
        std::function<bool()> constrained_update = nullptr,
        std::function<HessianType(const StateType &)> hess = nullptr,
        std::function<EqConstraintType(const StateType &)> eq_constraints = nullptr,
        std::function<IneqConstraintType(const StateType &)> ineq_constraints = nullptr,
        std::function<EqConstraintGradientType(const StateType &)> eq_constraints_grad = nullptr,
        std::function<IneqConstraintGradientType(const StateType &)> ineq_constraints_grad = nullptr)
        : func_(func),
          grad_(grad),
          residuals_(residuals ? residuals : [](const StateType &)
                         { return ResidualType(); }),
          residual_jacobian_(residual_jacobian ? residual_jacobian : [](const StateType &)
                                 { return ResidualJacobianType(); }),
          hess_(hess ? hess : [](const StateType &x)
                    { return HessianType::Zero(); }),
          eq_constraints_(eq_constraints ? eq_constraints : [](const StateType &x)
                              { return EqConstraintType(); }),
          ineq_constraints_(ineq_constraints ? ineq_constraints : [](const StateType &x)
                                { return IneqConstraintType(); }),
          eq_constraints_grad_(eq_constraints_grad ? eq_constraints_grad : [](const StateType &x)
                                   { return EqConstraintGradientType(); }),
          ineq_constraints_grad_(ineq_constraints_grad ? ineq_constraints_grad : [](const StateType &x)
                                     { return IneqConstraintGradientType(); }),
          unconstrained_update_(unconstrained_update ? unconstrained_update : []()
                                    { return true; }),
          constrained_update_(constrained_update ? constrained_update : []()
                                  { return true; }) {}

    Scalar evaluate(const StateType &x) const override
    {
        return func_(x);
    }

    GradientType gradient(const StateType &x) const override
    {
        return grad_(x);
    }

    HessianType hessian(const StateType &x) const override
    {
        return hess_(x);
    }

    EqConstraintType equalityConstraints(const StateType &x) const override
    {
        return eq_constraints_(x);
    }

    IneqConstraintType inequalityConstraints(const StateType &x) const override
    {
        return ineq_constraints_(x);
    }

    EqConstraintGradientType equalityConstraintsGradient(const StateType &x) const override
    {
        return eq_constraints_grad_(x);
    }

    IneqConstraintGradientType inequalityConstraintsGradient(const StateType &x) const override
    {
        return ineq_constraints_grad_(x);
    }

    ResidualType residuals(const StateType &x) const override
    {
        return residuals_(x);
    }

    ResidualJacobianType residualJacobian(const StateType &x) const override
    {
        return residual_jacobian_(x);
    }

    bool unconstrainedUpdate() const override
    {
        return unconstrained_update_();
    }

    bool constrainedUpdate() const override
    {
        return constrained_update_();
    }

private:
    std::function<Scalar(const StateType &)> func_;
    std::function<GradientType(const StateType &)> grad_;
    std::function<ResidualType(const StateType &)> residuals_;
    std::function<ResidualJacobianType(const StateType &)> residual_jacobian_;
    std::function<HessianType(const StateType &)> hess_;
    std::function<EqConstraintType(const StateType &)> eq_constraints_;
    std::function<IneqConstraintType(const StateType &)> ineq_constraints_;
    std::function<EqConstraintGradientType(const StateType &)> eq_constraints_grad_;
    std::function<IneqConstraintGradientType(const StateType &)> ineq_constraints_grad_;
    std::function<bool()> unconstrained_update_;
    std::function<bool()> constrained_update_;
};

#endif // CONCRETE_FUNCTION_H
