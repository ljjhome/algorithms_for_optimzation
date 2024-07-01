#ifndef MEASUREMENT_GROUP_H
#define MEASUREMENT_GROUP_H

#include <tuple>
#include "measurement_base.h"

template<typename Scalar, typename... Measurements>
class MeasurementGroup {
public:
    using MeasurementTuple = std::tuple<Measurements...>;
    static constexpr int TotalDim = (Measurements::getDimension() + ...);

    MeasurementGroup(const Measurements&... measurements)
        : measurements_(measurements...) {}

    MeasurementGroup(const MeasurementTuple& measurements)
        : measurements_(measurements) {}

    MeasurementGroup() : measurements_() {}

    const MeasurementTuple& getMeasurements() const {
        return measurements_;
    }

    Eigen::Matrix<Scalar, TotalDim, 1> getValues() const {
        Eigen::Matrix<Scalar, TotalDim, 1> values;
        int offset = 0;
        applyGetValue(values, offset);
        return values;
    }

    Eigen::Matrix<Scalar, TotalDim, TotalDim> getCovariances() const {
        Eigen::Matrix<Scalar, TotalDim, TotalDim> covariances;
        int offset = 0;
        applyGetCovariance(covariances, offset);
        return covariances;
    }

private:
    MeasurementTuple measurements_;

    void applyGetValue(Eigen::Matrix<Scalar, TotalDim, 1>& values, int& offset) const {
        forEachMeasurement([&](const auto& measurement) {
            constexpr int dim = measurement.getDimension();
            values.template segment<dim>(offset) = measurement.getValue();
            offset += dim;
        });
    }

    void applyGetCovariance(Eigen::Matrix<Scalar, TotalDim, TotalDim>& covariances, int& offset) const {
        forEachMeasurement([&](const auto& measurement) {
            constexpr int dim = measurement.getDimension();
            covariances.template block<dim, dim>(offset, offset) = measurement.getCovariance();
            offset += dim;
        });
    }

    template<typename Func, typename Tuple, std::size_t... Indices>
    void forEachMeasurementImpl(Func&& func, Tuple& tuple, std::index_sequence<Indices...>) const {
        (..., func(std::get<Indices>(tuple)));
    }

    template<typename Func, typename Tuple>
    void forEachMeasurement(Func&& func, Tuple& tuple) const {
        forEachMeasurementImpl(std::forward<Func>(func), tuple, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
    }
};

#endif // MEASUREMENT_GROUP_H
