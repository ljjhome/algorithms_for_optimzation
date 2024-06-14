#ifndef OPTIMIZER_INTERFACE_H
#define OPTIMIZER_INTERFACE_H

#include <Eigen/Core>
#include "function_interface.h"
#include "common/parameters/unified_optimizer_config.h"
#include <string>

template<typename Scalar, typename State,int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class OptimizerInterface {
public:
    using StateType = State;
    using FunctionType = FunctionInterface<Scalar, State,EqConstraints,IneqConstraints>;
    using GradientType = typename FunctionType::GradientType;

    virtual void optimize(StateType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) = 0;
    virtual bool isSuccess() const = 0;
    virtual std::string getReason() const = 0;
    virtual StateType getOptimizedVariables() const = 0;

    virtual ~OptimizerInterface() = default;
};

#endif // OPTIMIZER_INTERFACE_H
