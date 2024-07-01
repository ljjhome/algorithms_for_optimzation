#ifndef KALMAN_FILTER_BASE_H
#define KALMAN_FILTER_BASE_H

#include <memory>
#include "state.h"
#include "measurement_base.h"
#include "measurement_group.h"
#include "control_input_base.h"
#include "control_input_group.h"

template<typename State, typename Scalar>
class KalmanFilterBase {
public:
    virtual ~KalmanFilterBase() = default;

    virtual void predict(const std::shared_ptr<ControlInputGroup<Scalar>>& control_input_group) = 0;
    virtual void update(const std::shared_ptr<MeasurementGroup<Scalar>>& measurement_group) = 0;

    const State& getState() const {
        return state_;
    }

    void setState(const State& state) {
        state_ = state;
    }

    const Eigen::Matrix<Scalar, State::TotalDim, State::TotalDim>& getCovariance() const {
        return covariance_;
    }

    void setCovariance(const Eigen::Matrix<Scalar, State::TotalDim, State::TotalDim>& covariance) {
        covariance_ = covariance;
    }

protected:
    State state_;
    Eigen::Matrix<Scalar, State::TotalDim, State::TotalDim> covariance_;
};

#endif // KALMAN_FILTER_BASE_H
