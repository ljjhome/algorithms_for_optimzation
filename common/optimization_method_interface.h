#include <Eigen/Dense>

// optimization_method_interface.h
template<typename Scalar, int Rows, int Cols>
class OptimizationMethodInterface {
public:
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    virtual ~OptimizationMethodInterface() = default;
    virtual MatrixType getUpdateDirection(const MatrixType& x, const FunctionInterface<Scalar, Rows, Cols>& function) const = 0;
};
