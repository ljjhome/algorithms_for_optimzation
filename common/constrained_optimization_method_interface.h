#ifndef CONSTRAINED_OPTIMIZATION_METHOD_INTERFACE_H
#define CONSTRAINED_OPTIMIZATION_METHOD_INTERFACE_H

#include <Eigen/Core>
#include "function_interface.h"
#include "parameters/unified_optimizer_config.h"

template<typename Scalar, int Rows, int Cols, int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class ConstrainedOptimizationMethodInterface {
public:
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    using FunctionType = FunctionInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>;

    virtual void initialize(const MatrixType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) = 0;
    virtual Scalar getPenalty(const FunctionType& function, const MatrixType& x) const = 0;
    virtual void updateParameters(const MatrixType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) = 0;
    virtual bool hasConverged(const MatrixType& x, const FunctionType& function) const = 0;
    virtual void setTolerance(Scalar tol) = 0;

    virtual std::function<Scalar(const MatrixType&)> getAugmentedFunction(
        const FunctionType& function) const = 0;

    virtual std::function<MatrixType(const MatrixType&)> getAugmentedGradient(
        const FunctionType& function) const = 0;

    virtual ~ConstrainedOptimizationMethodInterface() = default;
};

#endif // CONSTRAINED_OPTIMIZATION_METHOD_INTERFACE_H
