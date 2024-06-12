#ifndef FUNCTION_INTERFACE_H
#define FUNCTION_INTERFACE_H

#include <Eigen/Core>

template<typename Scalar, int Rows, int Cols, int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class FunctionInterface {
public:
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    using HessianType = Eigen::Matrix<Scalar, Rows, Rows>;
    using EqConstraintType = Eigen::Matrix<Scalar, EqConstraints, 1>;
    using IneqConstraintType = Eigen::Matrix<Scalar, IneqConstraints, 1>;
    using EqConstraintGradientType = Eigen::Matrix<Scalar, EqConstraints, Rows>;
    using IneqConstraintGradientType = Eigen::Matrix<Scalar, IneqConstraints, Rows>;

    virtual Scalar evaluate(const MatrixType& x) const = 0;
    virtual MatrixType gradient(const MatrixType& x) const = 0;
    virtual HessianType hessian(const MatrixType& x) const = 0;

    virtual EqConstraintType equalityConstraints(const MatrixType& x) const = 0;
    virtual IneqConstraintType inequalityConstraints(const MatrixType& x) const = 0;
    virtual EqConstraintGradientType equalityConstraintsGradient(const MatrixType& x) const = 0;
    virtual IneqConstraintGradientType inequalityConstraintsGradient(const MatrixType& x) const = 0;

    virtual ~FunctionInterface() = default;
};

#endif // FUNCTION_INTERFACE_H
