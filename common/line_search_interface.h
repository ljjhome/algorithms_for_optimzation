#ifndef LINE_SEARCH_INTERFACE_H
#define LINE_SEARCH_INTERFACE_H

#include <Eigen/Core>
#include "common/function_interface.h"

template<typename Scalar, int Rows, int Cols>
class LineSearchInterface {
public:
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    using FunctionType = FunctionInterface<Scalar, Rows, Cols>;

    virtual Scalar search(const MatrixType& x, const MatrixType& dx, const FunctionInterface<Scalar, Rows, Cols>& f, int max_iter) const = 0;

    virtual ~LineSearchInterface() = default;
};

#endif // LINE_SEARCH_INTERFACE_H
