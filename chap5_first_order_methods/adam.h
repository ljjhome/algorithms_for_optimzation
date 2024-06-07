#pragma once
#include "common/optimization_method_interface.h"
#include "common/function_interface.h"


template<typename Scalar, int Rows, int Cols>
class Adam : public OptimizationMethodInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename OptimizationMethodInterface<Scalar, Rows, Cols>::MatrixType;

    Adam(Scalar learning_rate = 0.001, Scalar beta1 = 0.9, Scalar beta2 = 0.999, Scalar epsilon = 1e-8)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), m_(MatrixType::Zero()), v_(MatrixType::Zero()), t_(0) {}

    MatrixType getUpdateDirection(const MatrixType& x, const FunctionInterface<Scalar, Rows, Cols>& function) const override {
        MatrixType grad = function.gradient(x);
        t_ += 1;
        m_ = beta1_ * m_ + (1 - beta1_) * grad;
        v_ = beta2_ * v_ + (1 - beta2_) * grad.cwiseProduct(grad);
        MatrixType m_hat = m_ / (1 - std::pow(beta1_, t_));
        MatrixType v_hat = v_ / (1 - std::pow(beta2_, t_));
        return -learning_rate_ * m_hat.cwiseQuotient(v_hat.cwiseSqrt() + epsilon_);
    }

private:
    Scalar learning_rate_;
    Scalar beta1_;
    Scalar beta2_;
    Scalar epsilon_;
    mutable MatrixType m_;
    mutable MatrixType v_;
    mutable int t_;
};