// newton_method.h
#pragma once
#include "common/optimization_method_interface.h"
#include "common/function_interface.h"

template<typename Scalar, int Rows, int Cols>
class NewtonMethod : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;
    using HessianType = typename FunctionInterface<Scalar, Rows, Cols>::HessianType;

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionInterface<Scalar, Rows, Cols>& function) const override {
        // Retrieve the gradient and Hessian from the function interface
        MatrixType grad = function.gradient(x);
        HessianType H = function.hessian(x);
        // Solve for the Newton direction: H * p = -grad
        return -H.ldlt().solve(grad);
    }
};
