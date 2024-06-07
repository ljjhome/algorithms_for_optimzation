#pragma once
#include "common/optimization_method_interface.h"
#include "common/function_interface.h"
template<typename Scalar, int Rows, int Cols>
class NesterovMomentum : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;

    NesterovMomentum(Scalar alpha = 0.01, Scalar beta = 0.9)
        : alpha_(alpha), beta_(beta), v_(MatrixType::Zero()) {}

    void init(const MatrixType& x) {
        v_ = MatrixType::Zero(x.rows(), x.cols());
    }

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionInterface<Scalar, Rows, Cols>& function) const override {
        MatrixType grad = function.gradient(x + beta_ * v_);
        v_ = beta_ * v_ - alpha_ * grad;
        return v_;
    }

private:
    Scalar alpha_;
    Scalar beta_;
    mutable MatrixType v_;
};