#ifndef OPTIMIZATION_METHOD_INTERFACE_H
#define OPTIMIZATION_METHOD_INTERFACE_H

#include <Eigen/Core>
#include "common/function_interface.h"

template<typename Scalar, int Rows, int Cols>
class OptimizationMethodInterface {
public:
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    using FunctionType = FunctionInterface<Scalar, Rows, Cols>;

    virtual MatrixType getUpdateDirection(const MatrixType& x, const FunctionType& function) const = 0;

    virtual ~OptimizationMethodInterface() = default;
};

#endif // OPTIMIZATION_METHOD_INTERFACE_H
