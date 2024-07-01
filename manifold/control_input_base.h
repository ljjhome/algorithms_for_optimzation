#ifndef CONTROL_INPUT_BASE_H
#define CONTROL_INPUT_BASE_H

#include <Eigen/Dense>

template<typename Scalar, int Dim = Eigen::Dynamic>
class ControlInputBase {
public:
    virtual ~ControlInputBase() = default;

    static constexpr int getDimension() { return Dim; }
    virtual Eigen::Matrix<Scalar, Dim, 1> getValue() const = 0;
};

#endif // CONTROL_INPUT_BASE_H
