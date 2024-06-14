#ifndef CONSTRAINED_OPTIMIZATION_METHOD_INTERFACE_H
#define CONSTRAINED_OPTIMIZATION_METHOD_INTERFACE_H

#include <Eigen/Core>
#include "function_interface.h"
#include "common/parameters/unified_optimizer_config.h"

template<typename Scalar, typename State,int EqConstraints = Eigen::Dynamic, int IneqConstraints = Eigen::Dynamic>
class ConstrainedOptimizationMethodInterface {
public:
    using StateType = State;
    using FunctionType = FunctionInterface<Scalar, State,EqConstraints,IneqConstraints>;
    using GradientType = typename FunctionType::GradientType;

    virtual void initialize(const StateType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) = 0;
    virtual Scalar getPenalty(const FunctionType& function, const StateType& x) const = 0;
    virtual void updateParameters(const StateType& x, const FunctionType& function, const UnifiedOptimizerConfig& config) = 0;
    virtual bool hasConverged(const StateType& x, const FunctionType& function) const = 0;
    virtual void setTolerance(Scalar tol) = 0;

    virtual std::function<Scalar(const StateType&)> getAugmentedFunction(const FunctionType& function) const = 0;
    virtual std::function<GradientType(const StateType&)> getAugmentedGradient(const FunctionType& function) const = 0;

    virtual ~ConstrainedOptimizationMethodInterface() = default;
};

#endif // CONSTRAINED_OPTIMIZATION_METHOD_INTERFACE_H
