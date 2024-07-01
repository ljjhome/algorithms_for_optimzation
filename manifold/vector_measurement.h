#ifndef VECTOR_MEASUREMENT_H
#define VECTOR_MEASUREMENT_H

#include "measurement_base.h"

template<typename Scalar, int Dim = Eigen::Dynamic>
class VectorMeasurement : public MeasurementBase<Scalar, Dim> {
public:
    using VectorType = Eigen::Matrix<Scalar, Dim, 1>;
    using CovarianceType = Eigen::Matrix<Scalar, Dim, Dim>;

    VectorMeasurement(const VectorType& value, const CovarianceType& covariance)
        : value_(value), covariance_(covariance) {}

    Eigen::Matrix<Scalar, Dim, 1> getValue() const override {
        return value_;
    }

    Eigen::Matrix<Scalar, Dim, Dim> getCovariance() const override {
        return covariance_;
    }

    Eigen::Matrix<Scalar, Dim, Eigen::Dynamic> getMeasurementMatrix() const override {
        // Implement this based on your specific requirements
        return Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>::Identity(Dim, Dim);
    }

private:
    VectorType value_;
    CovarianceType covariance_;
};

#endif // VECTOR_MEASUREMENT_H
