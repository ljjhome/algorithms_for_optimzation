#ifndef ROTATION_MEASUREMENT_H
#define ROTATION_MEASUREMENT_H

#include "measurement_base.h"

template<typename Scalar, int Dim = 3>
class RotationMeasurement : public MeasurementBase<Scalar, Dim> {
public:
    using RotationType = Eigen::Matrix<Scalar, Dim, Dim>;
    using CovarianceType = Eigen::Matrix<Scalar, Dim, Dim>;

    RotationMeasurement(const RotationType& value, const CovarianceType& covariance)
        : value_(value), covariance_(covariance) {}

    Eigen::Matrix<Scalar, Dim, 1> getValue() const override {
        // Implement this based on your specific requirements
        return Eigen::Matrix<Scalar, Dim, 1>::Zero();
    }

    Eigen::Matrix<Scalar, Dim, Dim> getCovariance() const override {
        return covariance_;
    }

    Eigen::Matrix<Scalar, Dim, Eigen::Dynamic> getMeasurementMatrix() const override {
        // Implement this based on your specific requirements
        return Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>::Identity(Dim, Dim);
    }

private:
    RotationType value_;
    CovarianceType covariance_;
};

#endif // ROTATION_MEASUREMENT_H
