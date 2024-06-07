#pragma once
#include "common/optimization_method_interface.h"
#include "common/function_interface.h"

template<typename Scalar, int Rows, int Cols>
class Adadelta : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;

    Adadelta(Scalar decay_rate = 0.95, Scalar epsilon = 1e-6)
        : decay_rate_(decay_rate), epsilon_(epsilon), Eg_(MatrixType::Zero()), Ex_(MatrixType::Zero()) {}

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionInterface<Scalar, Rows, Cols>& function) const override {
        MatrixType grad = function.gradient(x);
        Eg_ = decay_rate_ * Eg_ + (1 - decay_rate_) * grad.cwiseProduct(grad);
        MatrixType update = -((Ex_ + epsilon_).cwiseSqrt().cwiseQuotient((Eg_ + epsilon_).cwiseSqrt())).cwiseProduct(grad);
        Ex_ = decay_rate_ * Ex_ + (1 - decay_rate_) * update.cwiseProduct(update);
        return update;
    }

private:
    Scalar decay_rate_;
    Scalar epsilon_;
    mutable MatrixType Eg_;
    mutable MatrixType Ex_;
};