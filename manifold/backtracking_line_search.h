#ifndef BACKTRACKING_LINE_SEARCH_H
#define BACKTRACKING_LINE_SEARCH_H

#include "line_search_interface.h"
#include <glog/logging.h>
template<typename Scalar, typename State,int ResidualDim = Eigen::Dynamic>
class BacktrackingLineSearch : public LineSearchInterface<Scalar, State,ResidualDim> {
public:
    using StateType = typename LineSearchInterface<Scalar, State,ResidualDim>::StateType;
    using FunctionType = typename LineSearchInterface<Scalar, State,ResidualDim>::FunctionType;
    using GradientType = typename LineSearchInterface<Scalar, State,ResidualDim>::GradientType;

    Scalar search(const StateType& x, const GradientType& dx, const FunctionType& f, int max_iter) const override {
        Scalar f0 = f.evaluate(x);
        Scalar alpha = 1.0;
        for (int i = 0; i < max_iter; ++i) {
            auto update = Eigen::Matrix<Scalar, State::TotalDim, 1>(alpha * dx);
            if (f.evaluate(x.boxPlus(update)) < f0) {
                return alpha;
            } else {
                alpha *= 0.05;
            }
        }
        // LOG(INFO)<<"alpha reach max iter";
        return alpha;
    }
};

#endif // BACKTRACKING_LINE_SEARCH_H
