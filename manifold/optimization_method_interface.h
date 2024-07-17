#ifndef OPTIMIZATION_METHOD_INTERFACE_H
#define OPTIMIZATION_METHOD_INTERFACE_H

#include <Eigen/Core>
#include "state.h"
#include "function_interface.h"

template<typename Scalar, typename State, int ResidualDim = Eigen::Dynamic>
class OptimizationMethodInterface {
public:
    using StateType = State;
    using FunctionType = FunctionInterface<Scalar, State, ResidualDim>;
    using GradientType = Eigen::Matrix<Scalar, State::TotalDim, 1>;

    virtual GradientType getUpdateDirection(const StateType& current_state, const FunctionType& function) const = 0;
    virtual void ResetParameters() = 0;
    virtual ~OptimizationMethodInterface() = default;
};

#endif // OPTIMIZATION_METHOD_INTERFACE_H
