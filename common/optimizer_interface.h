#ifndef OPTIMIZER_INTERFACE_H
#define OPTIMIZER_INTERFACE_H

#include <Eigen/Core>
#include "common/function_interface.h"
#include "common/parameters/unified_optimizer_config.h"

template<typename Scalar, int Rows, int Cols, int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class OptimizerInterface {
public:
    using MatrixType = Eigen::Matrix<Scalar, Rows, Cols>;
    using FunctionType = FunctionInterface<Scalar, Rows, Cols, EqConstraints, IneqConstraints>;

    virtual void optimize(MatrixType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) = 0;
    virtual bool isSuccess() const = 0;
    virtual std::string getReason() const = 0;
    virtual MatrixType getOptimizedVariables() const = 0;

    virtual ~OptimizerInterface() = default;
};

#endif // OPTIMIZER_INTERFACE_H
