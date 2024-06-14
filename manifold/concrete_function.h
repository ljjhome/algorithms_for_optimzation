#ifndef CONCRETE_FUNCTION_H
#define CONCRETE_FUNCTION_H

#include "function_interface.h"
#include <functional>

template<typename Scalar, typename State, int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class ConcreteFunction : public FunctionInterface<Scalar, State, EqConstraints, IneqConstraints> {
public:
    using StateType = State;
    using GradientType = typename FunctionInterface<Scalar, State, EqConstraints, IneqConstraints>::GradientType;
    using HessianType = typename FunctionInterface<Scalar, State, EqConstraints, IneqConstraints>::HessianType;
    using EqConstraintType = typename FunctionInterface<Scalar, State, EqConstraints, IneqConstraints>::EqConstraintType;
    using IneqConstraintType = typename FunctionInterface<Scalar, State, EqConstraints, IneqConstraints>::IneqConstraintType;
    using EqConstraintGradientType = typename FunctionInterface<Scalar, State, EqConstraints, IneqConstraints>::EqConstraintGradientType;
    using IneqConstraintGradientType = typename FunctionInterface<Scalar, State, EqConstraints, IneqConstraints>::IneqConstraintGradientType;

    ConcreteFunction(
        std::function<Scalar(const StateType&)> func,
        std::function<GradientType(const StateType&)> grad,
        std::function<HessianType(const StateType&)> hess = nullptr,
        std::function<EqConstraintType(const StateType&)> eq_constraints = nullptr,
        std::function<IneqConstraintType(const StateType&)> ineq_constraints = nullptr,
        std::function<EqConstraintGradientType(const StateType&)> eq_constraints_grad = nullptr,
        std::function<IneqConstraintGradientType(const StateType&)> ineq_constraints_grad = nullptr)
        : func_(func),
          grad_(grad),
          hess_(hess ? hess : [](const StateType& x) { return HessianType::Zero(); }),
          eq_constraints_(eq_constraints ? eq_constraints : [](const StateType& x) { return EqConstraintType(); }),
          ineq_constraints_(ineq_constraints ? ineq_constraints : [](const StateType& x) { return IneqConstraintType(); }),
          eq_constraints_grad_(eq_constraints_grad ? eq_constraints_grad : [](const StateType& x) { return EqConstraintGradientType(); }),
          ineq_constraints_grad_(ineq_constraints_grad ? ineq_constraints_grad : [](const StateType& x) { return IneqConstraintGradientType(); }) {}

    Scalar evaluate(const StateType& x) const override {
        return func_(x);
    }

    GradientType gradient(const StateType& x) const override {
        return grad_(x);
    }

    HessianType hessian(const StateType& x) const override {
        return hess_(x);
    }

    EqConstraintType equalityConstraints(const StateType& x) const override {
        return eq_constraints_(x);
    }

    IneqConstraintType inequalityConstraints(const StateType& x) const override {
        return ineq_constraints_(x);
    }

    EqConstraintGradientType equalityConstraintsGradient(const StateType& x) const override {
        return eq_constraints_grad_(x);
    }

    IneqConstraintGradientType inequalityConstraintsGradient(const StateType& x) const override {
        return ineq_constraints_grad_(x);
    }

private:
    std::function<Scalar(const StateType&)> func_;
    std::function<GradientType(const StateType&)> grad_;
    std::function<HessianType(const StateType&)> hess_;
    std::function<EqConstraintType(const StateType&)> eq_constraints_;
    std::function<IneqConstraintType(const StateType&)> ineq_constraints_;
    std::function<EqConstraintGradientType(const StateType&)> eq_constraints_grad_;
    std::function<IneqConstraintGradientType(const StateType&)> ineq_constraints_grad_;
};

#endif // CONCRETE_FUNCTION_H
