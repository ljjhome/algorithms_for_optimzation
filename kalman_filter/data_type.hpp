#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace rbl
{
    struct IMUData
    {
        double time;
        Eigen::Matrix<double, 3, 1> linear_acceleration;
        Eigen::Matrix<double, 3, 1> angular_velocity;
    };

    struct GPSData
    {
        double time;
        Eigen::Matrix<double, 3, 1> position_lla; // latitude longitude altitude in degree
        Eigen::Matrix<double, 3, 1> position_enu; // in meter
        Eigen::Matrix<double, 3, 1> orientation;
    };

    struct GroundTruthData
    {
        double time;
        Eigen::Matrix<double, 3, 1> true_position_enu;
        Eigen::Matrix<double, 3, 3> true_orientation;
        Eigen::Matrix<double, 3, 1> true_linear_acceleration;
        Eigen::Matrix<double, 3, 1> true_angular_velocity;
    };
    struct ConfigData
    {

    };

    
} // namespace rbl
