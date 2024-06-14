#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <math.h>

#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0

namespace rbl
{
    Eigen::Matrix<double, 3, 3> Hat(const Eigen::Matrix<double,3,1> &v)
    {
        Eigen::Matrix<double, 3, 3> h;
        h << 0.0, -v(2,0), v(1,0), v(2,0), 0.0, -v(0,0), -v(1,0), v(0,0), 0.0;
        return h;
    }
    Eigen::Matrix<double, 3, 3> Exp3x3(const Eigen::Matrix<double,3,1> &angle)
    {
        double angle_norm = angle.norm();
        Eigen::Matrix<double, 3, 3> Eye3 = Eigen::Matrix<double, 3, 3>::Identity();
        if (angle_norm < 1e-7)
        {
            return Eye3;
        }
        Eigen::Matrix<double, 3, 1> r_axis = angle / angle_norm;
        Eigen::Matrix<double, 3, 3> K;
        K  = Hat(r_axis);

        // Roderigous Tranformation
        return Eye3 + std::sin(angle_norm) * K + (1.0 - std::cos(angle_norm)) * K * K;
    }

    Eigen::Matrix<double, 3, 3> RightJacobian(const Eigen::Matrix<double,3,1> &v)
    {
        Eigen::Matrix<double, 3, 3> res = Eigen::Matrix<double, 3, 3>::Identity();
        double squaredNorm = v(0,0) * v(0,0) + v(1,0) * v(1,0)+ v(2,0) * v(2,0);
        double norm = std::sqrt(squaredNorm);
        if (norm < 1e-7)
        {
            return res;
        }
        res = Eigen::Matrix<double, 3, 3>::Identity() + (1 - std::cos(norm)) / squaredNorm * Hat(v) + (1 - std::sin(norm) / norm) / squaredNorm * Hat(v) * Hat(v);

        return res;
    }
    Eigen::Matrix<double, 3, 1> Log3x3(const Eigen::Matrix<double, 3, 3> &R)
    {
        double theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
        Eigen::Matrix<double, 3, 1> K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
        return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
    }
} // namespace rbl
