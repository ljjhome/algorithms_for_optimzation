#ifndef LINE_SEARCH_INTERFACE_H
#define LINE_SEARCH_INTERFACE_H

#include <Eigen/Core>
#include "function_interface.h"

template<typename Scalar, typename State>
class LineSearchInterface {
public:
    using StateType = State;
    using FunctionType = FunctionInterface<Scalar, State>;
    using GradientType = Eigen::Matrix<Scalar, State::TotalDim, 1>;

    virtual Scalar search(const StateType& x, const GradientType& dx, const FunctionType& f, int max_iter) const = 0;

    virtual ~LineSearchInterface() = default;
};

#endif // LINE_SEARCH_INTERFACE_H
