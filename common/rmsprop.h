#pragma once
#include "common/optimization_method_interface.h"
#include "common/function_interface.h"
template<typename Scalar, int Rows, int Cols>
class RMSprop : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;

    RMSprop(Scalar learning_rate = 0.001, Scalar decay_rate = 0.9, Scalar epsilon = 1e-8)
        : learning_rate_(learning_rate), decay_rate_(decay_rate), epsilon_(epsilon), cache_(MatrixType::Zero()) {}

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionInterface<Scalar, Rows, Cols>& function) const override {
        MatrixType grad = function.gradient(x);
        cache_ = decay_rate_ * cache_ + (1 - decay_rate_) * grad.cwiseProduct(grad);
        return -learning_rate_ * grad.cwiseQuotient((cache_ + epsilon_).cwiseSqrt());
    }

private:
    Scalar learning_rate_;
    Scalar decay_rate_;
    Scalar epsilon_;
    mutable MatrixType cache_;
};
