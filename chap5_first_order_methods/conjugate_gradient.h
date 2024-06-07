#pragma once
#include "common/optimization_method_interface.h"
#include "common/function_interface.h"
template<typename Scalar, int Rows, int Cols>
class ConjugateGradient : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;

    ConjugateGradient() : prev_grad_(MatrixType::Zero()), prev_direction_(MatrixType::Zero()), first_iteration_(true) {}

    MatrixType getUpdateDirection(const MatrixType& x, const MatrixType& grad) override {
        if (first_iteration_) {
            first_iteration_ = false;
            prev_grad_ = grad;
            prev_direction_ = -grad;
            return prev_direction_;
        }

        Scalar beta = grad.dot(grad) / prev_grad_.dot(prev_grad_);
        MatrixType direction = -grad + beta * prev_direction_;

        prev_grad_ = grad;
        prev_direction_ = direction;

        return direction;
    }

private:
    MatrixType prev_grad_;
    MatrixType prev_direction_;
    bool first_iteration_;
};
