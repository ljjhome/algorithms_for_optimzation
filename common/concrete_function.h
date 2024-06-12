#ifndef CONCRETE_FUNCTION_H
#define CONCRETE_FUNCTION_H

#include "common/function_interface.h"
#include <functional>

template<typename Scalar, int Rows, int Cols, int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class ConcreteFunction : public FunctionInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints> {
public:
    using MatrixType = typename FunctionInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>::MatrixType;
    using HessianType = typename FunctionInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>::HessianType;
    using EqConstraintType = typename FunctionInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>::EqConstraintType;
    using IneqConstraintType = typename FunctionInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>::IneqConstraintType;
    using EqConstraintGradientType = typename FunctionInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>::EqConstraintGradientType;
    using IneqConstraintGradientType = typename FunctionInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>::IneqConstraintGradientType;

    ConcreteFunction(
        std::function<Scalar(const MatrixType&)> func,
        std::function<MatrixType(const MatrixType&)> grad,
        std::function<HessianType(const MatrixType&)> hess = nullptr,
        std::function<EqConstraintType(const MatrixType&)> eq_constraints = nullptr,
        std::function<IneqConstraintType(const MatrixType&)> ineq_constraints = nullptr,
        std::function<EqConstraintGradientType(const MatrixType&)> eq_constraints_grad = nullptr,
        std::function<IneqConstraintGradientType(const MatrixType&)> ineq_constraints_grad = nullptr)
        : func_(func),
          grad_(grad),
          hess_(hess ? hess : [](const MatrixType& x) { return HessianType::Zero(); }),
          eq_constraints_(eq_constraints ? eq_constraints : [](const MatrixType& x) { return EqConstraintType(); }),
          ineq_constraints_(ineq_constraints ? ineq_constraints : [](const MatrixType& x) { return IneqConstraintType(); }),
          eq_constraints_grad_(eq_constraints_grad ? eq_constraints_grad : [](const MatrixType& x) { return EqConstraintGradientType(); }),
          ineq_constraints_grad_(ineq_constraints_grad ? ineq_constraints_grad : [](const MatrixType& x) { return IneqConstraintGradientType(); }) {}

    Scalar evaluate(const MatrixType& x) const override {
        return func_(x);
    }

    MatrixType gradient(const MatrixType& x) const override {
        return grad_(x);
    }

    HessianType hessian(const MatrixType& x) const override {
        return hess_(x);
    }

    EqConstraintType equalityConstraints(const MatrixType& x) const override {
        return eq_constraints_(x);
    }

    IneqConstraintType inequalityConstraints(const MatrixType& x) const override {
        return ineq_constraints_(x);
    }

    EqConstraintGradientType equalityConstraintsGradient(const MatrixType& x) const override {
        return eq_constraints_grad_(x);
    }

    IneqConstraintGradientType inequalityConstraintsGradient(const MatrixType& x) const override {
        return ineq_constraints_grad_(x);
    }

private:
    std::function<Scalar(const MatrixType&)> func_;
    std::function<MatrixType(const MatrixType&)> grad_;
    std::function<HessianType(const MatrixType&)> hess_;
    std::function<EqConstraintType(const MatrixType&)> eq_constraints_;
    std::function<IneqConstraintType(const MatrixType&)> ineq_constraints_;
    std::function<EqConstraintGradientType(const MatrixType&)> eq_constraints_grad_;
    std::function<IneqConstraintGradientType(const MatrixType&)> ineq_constraints_grad_;
};

#endif // CONCRETE_FUNCTION_H
