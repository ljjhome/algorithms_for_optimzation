#pragma once

#include "kalman_filter/data_type.hpp"
#include <string>
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
#include <deque>
#include <fstream>
#include <iostream>
#include "Geocentric/LocalCartesian.hpp"
#define kDegree2Radian (M_PI / 180.0)
namespace rbl
{
    class DataLoader
    {
        
    public:
        GeographicLib::LocalCartesian geo_converter_; // only support ENU
        void LLAToLocalNED(GPSData &gps_data)
        {
            // LLA -> ENU frame
            double enu_x, enu_y, enu_z;
            geo_converter_.Forward(gps_data.position_lla(0, 0),
                                   gps_data.position_lla(1, 0),
                                   gps_data.position_lla(2, 0),
                                   enu_x, enu_y, enu_z);
            gps_data.position_enu << enu_x, enu_y, enu_z;
        }
        Eigen::Vector3d LLAToLocalNED(const Eigen::Vector3d &lla)
        {
            Eigen::Vector3d enu;

            double enu_x, enu_y, enu_z;
            geo_converter_.Forward(lla.x(), lla.y(), lla.z(),
                                   enu_x, enu_y, enu_z);
            enu.x() = enu_x;
            enu.y() = enu_y;
            enu.z() = enu_z;

            return enu;
        }
        void LoadIMUData(const std::string &path, const int &skip, std::deque<IMUData> &res)
        {
            LOG(INFO) << "Read IMU data ...";
            std::string accel_file_path = path + "/accel-0.csv";
            std::string gyro_file_path = path + "/gyro-0.csv";
            std::string time_file_path = path + "/time.csv";

            std::ifstream accel_file(accel_file_path);
            std::ifstream gyro_file(gyro_file_path);
            std::ifstream time_file(time_file_path);

            if (!accel_file.is_open() || !time_file.is_open() ||
                !gyro_file.is_open())
            {
                LOG(INFO) << "Failure to open IMU file";
            }

            IMUData imu_data;
            res.clear();

            std::string accel_line;
            std::string ref_accel_line;
            std::string gyro_line;
            std::string ref_gyro_line;
            std::string time_line;
            std::string ref_pos_line;
            std::string ref_att_quat_line;
            std::string temp;

            for (int i = 0; i < skip; ++i)
            {
                std::getline(accel_file, temp);
                std::getline(gyro_file, temp);
                std::getline(time_file, temp);
            }

            while (std::getline(accel_file, accel_line) &&
                   std::getline(time_file, time_line) && std::getline(gyro_file, gyro_line))
            {
                imu_data.time = std::stod(time_line);

                std::stringstream ss;
                ss << accel_line;

                std::getline(ss, temp, ',');
                imu_data.linear_acceleration(0, 0) = std::stod(temp);
                std::getline(ss, temp, ',');
                imu_data.linear_acceleration(1, 0) = std::stod(temp);
                std::getline(ss, temp, ',');
                imu_data.linear_acceleration(2, 0) = std::stod(temp);

                ss.clear();
                ss << gyro_line;
                std::getline(ss, temp, ',');
                imu_data.angular_velocity(0, 0) = std::stod(temp) * kDegree2Radian;
                std::getline(ss, temp, ',');
                imu_data.angular_velocity(1, 0) = std::stod(temp) * kDegree2Radian;
                std::getline(ss, temp, ',');
                imu_data.angular_velocity(2, 0) = std::stod(temp) * kDegree2Radian;
                imu_data.linear_acceleration= imu_data.linear_acceleration + Eigen::Matrix<double,3,1>::Constant(0.1);
                imu_data.angular_velocity = imu_data.angular_velocity + Eigen::Matrix<double,3,1>::Constant(0.1);
                res.emplace_back(imu_data);
            }
            accel_file.close();
            gyro_file.close();
            LOG(INFO) << "Read IMU data successfully";
        }
        void LoadGPSData(const std::string &path, const int &skip, std::deque<GPSData> &res)
        {
            LOG(INFO) << "Read GPS data ...";
            std::string gps_file_path = path + "/gps-0.csv";
            std::string time_file_path = path + "/gps_time.csv";
            std::ifstream gps_file(gps_file_path, std::ios::in);
            std::ifstream gps_time_file(time_file_path, std::ios::in);

            if (!gps_file.is_open() || !gps_time_file.is_open())
            {
                LOG(FATAL) << "failure to open gps file";
            }

            GPSData gps_data;
            res.clear();

            std::string gps_data_line;
            std::string ref_gps_data_line;
            std::string gps_time_line;
            std::string temp;

            for (int i = 0; i < skip; ++i)
            {
                std::getline(gps_file, temp);
                std::getline(gps_time_file, temp);
            }
            bool is_first = true;
            while (std::getline(gps_file, gps_data_line) && std::getline(gps_time_file, gps_time_line))
            {
                gps_data.time = std::stod(gps_time_line);

                std::stringstream ssr_0;

                ssr_0 << gps_data_line;

                std::getline(ssr_0, temp, ',');
                gps_data.position_lla(0, 0) = std::stod(temp);

                std::getline(ssr_0, temp, ',');
                gps_data.position_lla(1, 0) = std::stod(temp);

                std::getline(ssr_0, temp, ',');
                gps_data.position_lla(2, 0) = std::stod(temp);

                if(is_first)
                {
                    // geo_converter_.Reset(gps_data.position_lla(0, 0), gps_data.position_lla(1, 0),gps_data.position_lla(2, 0));
                    geo_converter_.Reset(gps_data.position_lla(0, 0),gps_data.position_lla(1, 0), gps_data.position_lla(2, 0));
                    is_first = false;
                }
                LLAToLocalNED(gps_data);

                res.emplace_back(gps_data);
            }

            gps_time_file.close();
            gps_file.close();

            LOG(INFO) << "Read GPS data successfully";
        }

        void LoadGroundTruth(const std::string &path, const int &skip, std::deque<GroundTruthData> &res)
        {
            LOG(INFO) << "Read Groundtruth data ...";
            std::string ref_accel_file_path = path + "/ref_accel.csv";
            std::string ref_gyro_file_path = path + "/ref_gyro.csv";
            std::string time_file_path = path + "/time.csv";
            std::string ref_pos_file_path = path + "/ref_pos.csv";
            std::string ref_att_quat_file_path = path + "/ref_att_quat.csv";

            std::ifstream ref_accel_file(ref_accel_file_path, std::ios::in);

            std::ifstream ref_gyro_file(ref_gyro_file_path, std::ios::in);
            std::ifstream ref_att_quat_file(ref_att_quat_file_path, std::ios::in);
            std::ifstream ref_pos_file(ref_pos_file_path, std::ios::in);
            std::ifstream time_file(time_file_path, std::ios::in);

            if (!ref_accel_file.is_open() || !ref_gyro_file.is_open() || !time_file.is_open() || !ref_att_quat_file.is_open() || !ref_pos_file.is_open())
            {
                LOG(FATAL) << "Failure to open IMU file";
            }

            GroundTruthData gtdata;
            res.clear();

            std::string accel_line;
            std::string ref_accel_line;
            std::string gyro_line;
            std::string ref_gyro_line;
            std::string ref_pos_line;
            std::string ref_att_quat_line;
            std::string time_line;
            std::string temp;

            for (int i = 0; i < skip; ++i)
            {
                std::getline(ref_accel_file, temp);
                std::getline(ref_gyro_file, temp);
                std::getline(time_file, temp);
                std::getline(ref_pos_file, temp);
                std::getline(ref_att_quat_file, temp);
            }

            while (std::getline(ref_accel_file, ref_accel_line) &&
                   std::getline(ref_gyro_file, ref_gyro_line) &&
                   std::getline(time_file, time_line) &&
                   std::getline(ref_att_quat_file, ref_att_quat_line) &&
                   std::getline(ref_pos_file, ref_pos_line))
            {
                gtdata.time = std::stod(time_line);

                std::stringstream ss;

                ss << ref_accel_line;
                std::getline(ss, temp, ',');
                gtdata.true_linear_acceleration.x() = std::stod(temp);
                std::getline(ss, temp, ',');
                gtdata.true_linear_acceleration.y() = std::stod(temp);
                std::getline(ss, temp, ',');
                gtdata.true_linear_acceleration.z() = std::stod(temp);

                ss.clear();
                ss << ref_gyro_line;
                std::getline(ss, temp, ',');
                gtdata.true_angular_velocity.x() = std::stod(temp) * kDegree2Radian;
                std::getline(ss, temp, ',');
                gtdata.true_angular_velocity.y() = std::stod(temp) * kDegree2Radian;
                std::getline(ss, temp, ',');
                gtdata.true_angular_velocity.z() = std::stod(temp) * kDegree2Radian;

                ss.clear();
                ss << ref_pos_line;
                std::getline(ss, temp, ',');
                gtdata.true_position_enu.x() = std::stod(temp);
                std::getline(ss, temp, ',');
                gtdata.true_position_enu.y() = std::stod(temp);
                std::getline(ss, temp, ',');
                gtdata.true_position_enu.z() = std::stod(temp);

                ss.clear();
                ss << ref_att_quat_line;
                std::getline(ss, temp, ',');
                Eigen::Quaterniond qin;
                qin.w() = std::stod(temp);
                std::getline(ss, temp, ',');
                qin.x() = std::stod(temp);
                std::getline(ss, temp, ',');
                qin.y() = std::stod(temp);
                std::getline(ss, temp, ',');
                qin.z() = std::stod(temp);
                gtdata.true_orientation = qin.toRotationMatrix();

                res.emplace_back(gtdata);
            }

            LOG(INFO) << "Read IMU data successfully";
        }
    };
} // namespace rbl
