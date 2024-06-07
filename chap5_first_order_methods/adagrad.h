#pragma once
#include "common/optimization_method_interface.h"
#include "common/function_interface.h"

template<typename Scalar, int Rows, int Cols>
class Adagrad : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;

    Adagrad(Scalar learning_rate = 0.01, Scalar epsilon = 1e-8)
        : learning_rate_(learning_rate), epsilon_(epsilon), cache_(MatrixType::Zero()) {}

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionInterface<Scalar, Rows, Cols>& function) const override {
        MatrixType grad = function.gradient(x);
        cache_ += grad.cwiseProduct(grad);
        return -learning_rate_ * grad.cwiseQuotient((cache_ + epsilon_).cwiseSqrt());
    }

private:
    Scalar learning_rate_;
    Scalar epsilon_;
    mutable MatrixType cache_;
};