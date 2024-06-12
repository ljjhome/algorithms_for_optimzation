#ifndef ADADELTA_H
#define ADADELTA_H

#include "common/optimization_method_interface.h"

template<typename Scalar, int Rows, int Cols>
class AdaDelta : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;
    using FunctionType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::FunctionType;

    AdaDelta(Scalar rho = 0.95, Scalar epsilon = 1e-6)
        : rho_(rho), epsilon_(epsilon) {
        Eg_.setZero();
        Edx_.setZero();
    }

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionType& function) const override {
        MatrixType grad = function.gradient(x);
        Eg_ = rho_ * Eg_ + (1 - rho_) * grad.array().square().matrix();
        MatrixType delta_x = -(grad.array() * (Edx_.array() + epsilon_).sqrt() / (Eg_.array() + epsilon_).sqrt()).matrix();
        Edx_ = rho_ * Edx_ + (1 - rho_) * delta_x.array().square().matrix();
        return delta_x;
    }

private:
    Scalar rho_;
    Scalar epsilon_;
    mutable MatrixType Eg_;
    mutable MatrixType Edx_;
};

#endif // ADADELTA_H
