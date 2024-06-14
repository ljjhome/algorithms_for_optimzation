#include "ekf03c.hpp"
#include "kalman_filter/data_type.hpp"
#include "kalman_filter/data_loader.hpp"
#include <glog/logging.h>
#include <deque>
#include <iomanip>
void SaveTUMPose(std::ofstream &ofs, const double &q, const Eigen::Vector3d& axis,
                 const Eigen::Vector3d &t, double timestamp)
{
    ofs << std::fixed << std::setprecision(10) << timestamp << " " << t.x() << " " << t.y() << " " << t.z() << " "
        << q<< " " << axis.x() << " " << axis.y() << " " << axis.z() << std::endl;
}

int main(int argc, char **argv)
{
    // log setting
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_alsologtostderr = true;
    google::SetLogDestination(google::INFO, "../data/log/");

    // load data
    LOG(INFO) << "start load data";
    std::deque<rbl::GPSData> gps_queue;
    std::deque<rbl::IMUData> imu_queue;
    std::deque<rbl::GroundTruthData> gt_queue;
    rbl::DataLoader loader;
    std::string file_path = "/home/ljj/autoad/workspace/src/auto_ad/modules/auto_ad_localization/locinone/data";
    std::string file_path2 = file_path+"/raw_data";
    loader.LoadGPSData(file_path2, 1, gps_queue);
    loader.LoadIMUData(file_path2, 1, imu_queue);
    loader.LoadGroundTruth(file_path2,1,gt_queue);
    // ekf
    rbl::EKF03C ekf03;

    // initialize parameters
    Eigen::Matrix<double, 15, 15> P_init = Eigen::Matrix<double, 15, 15>::Identity()*0.1;
    ekf03.SetInitialCovP(P_init);
    Eigen::Matrix<double, 6, 6> Q_init = Eigen::Matrix<double, 6, 6>::Identity()*5;
    ekf03.SetCQ(Q_init);
    Eigen::Matrix<double, 3, 3> CR_init = Eigen::Matrix<double, 3, 3>::Identity()*50;
    ekf03.SetCR(CR_init);

    Eigen::Matrix<double, 3, 1> g_init{0, 0, 9.8};
    ekf03.SetGravity(g_init);

    // initialize state
    rbl::GPSData cur_gps = gps_queue.front();
    rbl::IMUData cur_imu = imu_queue.front();
    rbl::GroundTruthData cur_gt = gt_queue.front();

    ekf03.SetInitialStateP(cur_gps.position_enu);

    ekf03.SetInitialStateV(Eigen::Matrix<double, 3, 1>::Constant(0.0));

    ekf03.SetInitialStateR(Eigen::Matrix<double, 3, 3>::Identity());

    // pop the first elements in queues
    gps_queue.pop_front();
    imu_queue.pop_front();
    gt_queue.pop_front();


    double old_time = cur_imu.time;

    // out put file
    std::ofstream gt_file(file_path + "/gt.txt", std::ios::trunc);
    std::ofstream fused_file(file_path + "/fused.txt", std::ios::trunc);
    std::ofstream measured_file(file_path + "/gps_measurement.txt", std::ios::trunc);
    // loop over all data
    while (!gps_queue.empty() && !imu_queue.empty() )
    {
        cur_gps = gps_queue.front();
        cur_imu = imu_queue.front();
        cur_gt = gt_queue.front();
        
        if (cur_imu.time < cur_gps.time) // do prediction step
        {
            double dt = cur_imu.time - old_time;
            ekf03.SetInput(cur_imu.angular_velocity, cur_imu.linear_acceleration);
            ekf03.Predict(dt);
            imu_queue.pop_front();
            gt_queue.pop_front();
            old_time = cur_imu.time;
        }
        else // do update step
        {
            ekf03.SetMeasurements(cur_gps.position_enu);
            ekf03.Update();
            gps_queue.pop_front();


            // save result
            Eigen::AngleAxisd fused_q(ekf03.GetStateR());
            SaveTUMPose(fused_file, fused_q.angle(),fused_q.axis(),
                        ekf03.GetStateP(), cur_imu.time);

            SaveTUMPose(measured_file, 0, Eigen::Vector3d::UnitZ(),
                        cur_gps.position_enu, cur_gps.time);

            Eigen::AngleAxisd gt_q(cur_gt.true_orientation);
            SaveTUMPose(gt_file, gt_q.angle(),gt_q.axis(),
                        loader.LLAToLocalNED(cur_gt.true_position_enu), cur_gt.time);
        }
    }

    return 0;
}