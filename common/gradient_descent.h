#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include "common/optimization_method_interface.h"

template<typename Scalar, int Rows, int Cols>
class GradientDescent : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;
    using FunctionType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::FunctionType;

    GradientDescent(Scalar learning_rate = 0.1)
        : learning_rate_(learning_rate) {}

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionType& function) const override {
        return -learning_rate_ * function.gradient(x);
    }

private:
    Scalar learning_rate_;
};

#endif // GRADIENT_DESCENT_H
