#pragma once
#include "common/optimization_method_interface.h"
#include "common/function_interface.h"
template<typename Scalar, int Rows, int Cols>
class GradientDescent : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionInterface<Scalar, Rows, Cols>& function) const override {
        MatrixType grad = function.gradient(x);
        return -grad;
    }
};