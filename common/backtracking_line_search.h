#pragma once
#include "common/line_search_interface.h"


template<typename Scalar, int Rows, int Cols>
class BacktrackingLineSearch : public LineSearchInterface<Scalar, Rows, Cols> {
public:
    using MatrixType = typename LineSearchInterface<Scalar, Rows, Cols>::MatrixType;

    Scalar search(const MatrixType& x, const MatrixType& dx, const FunctionInterface<Scalar, Rows, Cols>& f, int max_iter) const override {
        Scalar f0 = f.evaluate(x);
        Scalar alpha = 1.0;
        for (int i = 0; i < max_iter; ++i) {
            if (f.evaluate(x + alpha * dx) < f0) {
                return alpha;
            } else {
                alpha *= 0.5;
            }
        }
        return alpha;
    }
};
