#ifndef MEASUREMENT_BASE_H
#define MEASUREMENT_BASE_H

#include <Eigen/Dense>

template<typename Scalar, int Dim = Eigen::Dynamic>
class MeasurementBase {
public:
    virtual ~MeasurementBase() = default;

    virtual int getDimension() const { return Dim; }
    virtual Eigen::Matrix<Scalar, Dim, 1> getValue() const = 0;
    virtual Eigen::Matrix<Scalar, Dim, Dim> getCovariance() const = 0;
    virtual Eigen::Matrix<Scalar, Dim, Eigen::Dynamic> getMeasurementMatrix() const = 0;
};

#endif // MEASUREMENT_BASE_H
