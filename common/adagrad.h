#ifndef ADAGRAD_H
#define ADAGRAD_H

#include "common/optimization_method_interface.h"

template<typename Scalar, int Rows, int Cols>
class AdaGrad : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;
    using FunctionType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::FunctionType;

    AdaGrad(Scalar learning_rate = 0.01, Scalar epsilon = 1e-8)
        : learning_rate_(learning_rate), epsilon_(epsilon) {
        G_.setZero();
    }

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionType& function) const override {
        MatrixType grad = function.gradient(x);
        G_ += grad.array().square().matrix();
        return -(learning_rate_ / (G_.array().sqrt() + epsilon_).matrix()).array() * grad.array();
    }

private:
    Scalar learning_rate_;
    Scalar epsilon_;
    mutable MatrixType G_;
};

#endif // ADAGRAD_H
