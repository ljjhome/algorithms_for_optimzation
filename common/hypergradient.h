#pragma once
#include "common/optimization_method_interface.h"
#include "common/function_interface.h"


template<typename Scalar, int Rows, int Cols>
class HypergradientDescent : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;

    HypergradientDescent(Scalar initial_learning_rate = 0.01, Scalar hypergradient_rate = 0.001)
        : learning_rate_(initial_learning_rate), hypergradient_rate_(hypergradient_rate) {}

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionInterface<Scalar, Rows, Cols>& function) const override {
        MatrixType grad = function.gradient(x);
        learning_rate_ -= hypergradient_rate_ * grad.squaredNorm();
        return -learning_rate_ * grad;
    }

private:
    mutable Scalar learning_rate_;
    Scalar hypergradient_rate_;
};