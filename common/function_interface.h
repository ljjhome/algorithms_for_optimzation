#pragma once
#include <functional>
#include <Eigen/Dense>

// function_interface.h
template<typename Scalar, int Rows, int Cols>
class FunctionInterface {
public:
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    using HessianType = Eigen::Matrix<Scalar, Rows, Rows>;
    virtual ~FunctionInterface() = default;
    virtual Scalar evaluate(const MatrixType& x) const = 0;
    virtual MatrixType gradient(const MatrixType& x) const = 0;
    virtual HessianType hessian(const MatrixType& x) const = 0;
};


// concrete_function.h
template<typename Scalar, int Rows, int Cols>
class ConcreteFunction : public FunctionInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename FunctionInterface<Scalar, Rows, Cols>::MatrixType;
    using HessianType = typename FunctionInterface<Scalar, Rows, Cols>::HessianType;

    ConcreteFunction(std::function<Scalar(const MatrixType&)> func,
                     std::function<MatrixType(const MatrixType&)> grad,
                     std::function<HessianType(const MatrixType&)> hessian)
        : func_(func), grad_(grad), hessian_(hessian) {}

    Scalar evaluate(const MatrixType& x) const override {
        return func_(x);
    }

    MatrixType gradient(const MatrixType& x) const override {
        return grad_(x);
    }

    HessianType hessian(const MatrixType& x) const override {
        return hessian_(x);
    }

private:
    std::function<Scalar(const MatrixType&)> func_;
    std::function<MatrixType(const MatrixType&)> grad_;
    std::function<HessianType(const MatrixType&)> hessian_;
};
