#pragma once
#include <Eigen/Dense>
#include "common/function_interface.h"
#include "common/line_search_interface.h"
template<typename Scalar, int Rows, int Cols>
class LineSearchInterface {
public:
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    virtual ~LineSearchInterface() = default;
    virtual Scalar search(const MatrixType& x, const MatrixType& dx, const FunctionInterface<Scalar, Rows, Cols>& f, int max_iter) const = 0;
};