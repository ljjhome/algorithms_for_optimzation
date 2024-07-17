#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "manifold/math.hpp"
#include <glog/logging.h>
#include "manifold/concrete_function.h"
#include "manifold/unconstrained_optimizer.h"
#include "manifold/backtracking_line_search.h"
#include "manifold/gradient_descent.h"
#include "manifold/conjugate_gradient.h"
#include "manifold/momentum_method.h"
#include "manifold/adam.h"
#include "manifold/penalty_method.h"
#include "manifold/augmented_lagrangian_method.h"
#include "manifold/constrained_optimizer.h"
#include "common/parameters/unified_optimizer_config.h"
#include "manifold/vector_state_component.h"
#include "manifold/rotation_state_component.h"
#include "manifold/gauss_newton_method.h"
#include <Eigen/Cholesky>
namespace rbl
{
    class EKF03U
    {
    public:
        EKF03U()
        {
            config_.loadFromYaml("/home/ljj/autoad/workspace/src/auto_ad/modules/auto_ad_localization/locinone/config/OptimizerConfig/optimizer_config.yaml");
        }
        ~EKF03U() = default;
        void SetInput(const Eigen::Matrix<double, 3, 1> &omega, const Eigen::Matrix<double, 3, 1> &a)
        {
            omega_ = omega;
            a_ = a;
        }
        void SetInitialCovP(const Eigen::Matrix<double, 15, 15> &Pin)
        {
            P_ = Pin;
        }
        void SetCQ(const Eigen::Matrix<double, 6, 6> &Qin)
        {
            C_Q_ = Qin;
        }
        void SetCR(const Eigen::Matrix<double, 3, 3> &Rin)
        {
            C_R_ = Rin;
        }
        void SetInitialStateP(const Eigen::Matrix<double, 3, 1> &state_p_in)
        {
            p_G_N_ = state_p_in;
        }
        void SetInitialStateR(const Eigen::Matrix<double, 3, 3> &state_R_in)
        {
            R_G_N_ = state_R_in;
        }
        void SetInitialStateV(const Eigen::Matrix<double, 3, 1> &state_v_in)
        {
            v_G_N_ = state_v_in;
        }
        void SetGravity(const Eigen::Matrix<double, 3, 1> &g)
        {
            g_G_ = g;
        }
        void Predict(double dt)
        {
            StateProp(dt);
            CovarianceProp(dt);
        }
        void StateProp(double dt)
        {
            current_R_ = R_G_N_;
            current_p_ = p_G_N_;
            current_v_ = v_G_N_;
            current_b_a_ = b_a_;
            current_b_omega_ = b_omega_;
            R_G_N_ = current_R_ * Exp3x3((omega_ - current_b_omega_) * dt);
            v_G_N_ = current_v_ + (current_R_ * (a_ - current_b_a_) + g_G_) * dt;
            p_G_N_ = current_p_ + current_v_ * dt;
        }
        Eigen::Matrix<double, 3, 1> GetStateP()
        {
            return p_G_N_;
        }
        Eigen::Matrix<double, 3, 1> GetStateV()
        {
            return v_G_N_;
        }
        Eigen::Matrix<double, 3, 3> GetStateR()
        {
            return R_G_N_;
        }
        void CovarianceProp(double dt)
        {
            /**
             * 旋转      平移         平移速度      加速度偏置      角速度偏置
             * R         p            v             b_a_            b_omega_
             */
            // 旋转对旋转
            F_x_.block<3, 3>(0, 0) = Exp3x3(-(omega_ - b_omega_) * dt);

            // 旋转对角速度偏置
            auto Jr = RightJacobian((omega_ - b_omega_) * dt);
            F_x_.block<3, 3>(0, 12) = -Jr * dt;

            // 平移对平移
            F_x_.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity();

            // 平移对速度
            F_x_.block<3, 3>(3, 6) = Eigen::Matrix<double, 3, 3>::Identity() * dt;

            // 速度对旋转
            F_x_.block<3, 3>(6, 0) = -current_R_ * Hat(a_ - b_a_) * dt;

            // 速度对速度
            F_x_.block<3, 3>(6, 6) = Eigen::Matrix<double, 3, 3>::Identity();

            // 速度对加速度偏置
            F_x_.block<3, 3>(6, 9) = -current_R_ * dt;

            // 加速度偏置，对加速度偏置
            F_x_.block<3, 3>(9, 9) = Eigen::Matrix<double, 3, 3>::Identity();

            // 角速度偏置对角速度偏置
            F_x_.block<3, 3>(12, 12) = Eigen::Matrix<double, 3, 3>::Identity();

            //
            // 旋转对角速度过程误差
            Jr = RightJacobian((omega_ - b_omega_) * dt);

            F_w_.block<3, 3>(0, 0) = -Jr * dt;

            // 速度对加速度过程误差
            F_w_.block<3, 3>(6, 3) = -current_R_ * dt;
            // // 加速度偏置，对偏置传递误差
            // F_w_.block<3, 3>(9, 6) = Eigen::Matrix<double, 3, 3>::Identity() * dt;
            // // 角速度偏置，对偏置传递误差
            // F_w_.block<3, 3>(12, 9) = Eigen::Matrix<double, 3, 3>::Identity() * dt;

            P_ = F_x_ * P_ * F_x_.transpose() + F_w_ * C_Q_ * F_w_.transpose();
        }

        void SetMeasurements(const Eigen::Matrix<double, 3, 1> &p)
        {
            p_m_ = p;
        }

        void Update()
        {
            auto R_predict = R_G_N_;
            auto p_predict = p_G_N_;
            auto v_predict = v_G_N_;
            auto b_a_predict = b_a_;
            auto b_omega_predict = b_omega_;
            auto P_pred = P_;
            auto R_inloop = R_G_N_;
            auto p_inloop = p_G_N_;
            auto v_inloop = v_G_N_;
            auto b_a_inloop = b_a_;
            auto b_omega_inloop = b_omega_;
            // LOG(INFO) << "before pose: " << "\n"
            //           << p_G_N_;
            // LOG(INFO) << "before velocity: \n"
            //           << v_G_N_;
            H_.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity();
            for (int i = 0; i < total_iter_num_; i++)
            {

                // LOG(INFO) << "P: \n"
                //           << P_;
                // LOG(INFO) << "H_:\n"
                //           << H_;
                // LOG(INFO) << "R: \n"
                //           << C_R_;
                auto K = P_ * H_.transpose() * (C_R_ + H_ * P_ * H_.transpose()).inverse();
                Eigen::Matrix<double, 15, 1> dX;
                Eigen::Matrix<double, 3, 1> dY;
                dY << (p_inloop - p_m_);
                auto Eye15x15 = Eigen::Matrix<double, 15, 15>::Identity();
                Eigen::Matrix<double, 15, 1> dY2;
                dY2.block<3, 1>(0, 0) = Log3x3(R_predict.inverse() * R_inloop);
                dY2.block<3, 1>(3, 0) = p_inloop - p_predict;
                dY2.block<3, 1>(6, 0) = v_inloop - v_predict;
                dY2.block<3, 1>(9, 0) = b_a_inloop - b_a_predict;
                dY2.block<3, 1>(12, 0) = b_omega_inloop - b_omega_predict;

                // Eigen::Matrix<double,15,15> HRHP = Eigen::Matrix<double,15,15>::Zero();
                // HRHP = H_.transpose() * C_R_.inverse() * H_ + P_.inverse();

                // HRHP = Eye15x15 - K * H_;
                // // LOG(INFO)<<"in kalm: \n"<<HRHP;
                
                // LOG(INFO) << "Y:\n"
                //           << dY;
                // LOG(INFO) << "K:\n"
                //           << K;
                dX = -(Eye15x15 - K * H_) * dY2 - K * dY;
                // LOG(INFO)<<"I-KH: \n"<< (Eye15x15 - K * H_);
                LOG(INFO) << "dX:\n"
                          << dX;
                R_inloop = R_inloop * Exp3x3(dX.block<3, 1>(0, 0));
                p_inloop = p_inloop + dX.block<3, 1>(3, 0);
                v_inloop = v_inloop + dX.block<3, 1>(6, 0);
                b_a_inloop = b_a_inloop + dX.block<3, 1>(9, 0);
                b_omega_inloop = b_omega_inloop + dX.block<3, 1>(9, 0);
                P_ = (Eye15x15 - K * H_) * P_;
            }

            // here is the optimizer
            R_predict = R_G_N_;
            p_predict = p_G_N_;
            v_predict = v_G_N_;
            b_a_predict = b_a_;
            b_omega_predict = b_omega_;

            using Scalar = double;
            using Statetype = State<Scalar, RotationStateComponent<Scalar>, VectorStateComponent<Scalar, 3>, VectorStateComponent<Scalar, 3>, VectorStateComponent<Scalar, 3>, VectorStateComponent<Scalar, 3>>;

            VectorStateComponent<Scalar, 3> p_component(p_G_N_);
            VectorStateComponent<Scalar, 3> v_component(v_G_N_);
            RotationStateComponent<Scalar> R_component(R_G_N_);
            VectorStateComponent<Scalar, 3> b_a_component(b_a_);
            VectorStateComponent<Scalar, 3> b_omega_component(b_omega_);
            Statetype state(R_component, p_component, v_component, b_a_component, b_omega_component);

            // auto C_R_inv = C_R_.inverse();
            // auto P_inv = P_pred.inverse();

            // auto P_inv_sqrt = P_inv.llt().matrixL();
            // auto C_R_inv_sqrt = C_R_inv.llt().matrixL();

            // Compute the inverse of the covariance matrices
            auto C_R_inv = C_R_.inverse();
            auto P_inv = P_pred.inverse();

            Eigen::Matrix<double,18,18> Cov_total_inv = Eigen::Matrix<double,18,18>::Zero();
            Cov_total_inv.block<15,15>(0,0) = P_inv;
            Cov_total_inv.block<3,3>(15,15) = C_R_inv;
            // LOG(INFO)<<"Cov_total_inv: \n"<<Cov_total_inv;
            // Compute the Cholesky decomposition
            Eigen::LLT<Eigen::Matrix<double, 15, 15>> P_inv_llt(P_inv);
            Eigen::LLT<Eigen::Matrix<double, 3, 3>> C_R_inv_llt(C_R_inv);
            Eigen::LLT<Eigen::Matrix<double, 18, 18>> Cov_total_inv_llt(Cov_total_inv);
            
            // Check if the decomposition was successful
            if (P_inv_llt.info() != Eigen::Success)
            {
                LOG(ERROR) << "Cholesky decomposition of P_inv failed!";
            }
            if (C_R_inv_llt.info() != Eigen::Success)
            {
                LOG(ERROR) << "Cholesky decomposition of C_R_inv failed!";
            }

            // Compute the square root matrices
            Eigen::Matrix<double, 15, 15> P_inv_sqrt = P_inv_llt.matrixL().transpose();
            Eigen::Matrix<double, 3, 3> C_R_inv_sqrt = C_R_inv_llt.matrixL().transpose();
            Eigen::Matrix<double, 18, 18> Cov_total_inv_sqrt = Cov_total_inv_llt.matrixL().transpose();
            // LOG(INFO)<<"P_inv_llt:\n "<<P_inv_sqrt;
            // LOG(INFO)<<"C_R_inv_llt:\n "<<C_R_inv_sqrt;
            // LOG(INFO)<<"Cov_total_inv_llt:\n "<<Cov_total_inv_sqrt;
            // // Log the results
            // LOG(INFO) << "P_inv: \n"
            //           << P_inv;
            // LOG(INFO) << "P_inv_sqrt: \n"
            //           << P_inv_sqrt;
            Eigen::Matrix<double,15,15> P_re = P_inv_sqrt.transpose()*P_inv_sqrt;
            // LOG(INFO)<<"P_re: \n"<<P_re;
            // LOG(INFO) << "C_R_inv: \n"
            //           << C_R_inv;
            // LOG(INFO) << "C_R_inv_sqrt: \n"
            //           << C_R_inv_sqrt;

            auto measurement_function = [&, this](const Statetype &state) -> Scalar
            {
                auto R = std::get<0>(state.getComponents()).getRotation();
                auto p = std::get<1>(state.getComponents()).getVector();
                auto v = std::get<2>(state.getComponents()).getVector();
                auto ba = std::get<3>(state.getComponents()).getVector();
                auto bomega = std::get<4>(state.getComponents()).getVector();
                Eigen::Matrix<double, 3, 1> p_merror;
                p_merror << (p - p_m_);
                Eigen::Matrix<double, 15, 1> dYY;
                dYY.block<3, 1>(0, 0) = Log3x3(R_predict.inverse() * R);
                dYY.block<3, 1>(3, 0) = p - p_predict;
                dYY.block<3, 1>(6, 0) = v - v_predict;
                dYY.block<3, 1>(9, 0) = ba - b_a_predict;
                dYY.block<3, 1>(12, 0) = bomega - b_omega_predict;
                auto cost = p_merror.transpose() * C_R_inv * p_merror + dYY.transpose() * P_inv * dYY;
                return cost.value();
            };

            auto measurement_gradient = [&, this](const Statetype &state) -> Eigen::Matrix<Scalar, 15, 1>
            {
                auto R = std::get<0>(state.getComponents()).getRotation();
                auto p = std::get<1>(state.getComponents()).getVector();
                auto v = std::get<2>(state.getComponents()).getVector();
                auto ba = std::get<3>(state.getComponents()).getVector();
                auto bomega = std::get<4>(state.getComponents()).getVector();
                Eigen::Matrix<double, 3, 1> Hx_z = p - p_m_;
                Eigen::Matrix<double, 15, 1> x_x0;
                x_x0.block<3, 1>(0, 0) = Log3x3(R_predict.inverse() * R);
                x_x0.block<3, 1>(3, 0) = p - p_predict;
                x_x0.block<3, 1>(6, 0) = v - v_predict;
                x_x0.block<3, 1>(9, 0) = ba - b_a_predict;
                x_x0.block<3, 1>(12, 0) = bomega - b_omega_predict;
                Eigen::Matrix<Scalar, 15, 1> grad = 2 * H_.transpose() * C_R_inv * Hx_z + 2 * P_inv * x_x0;
                return grad;
            };

            auto residual_function = [&, this](const Statetype &state) -> Eigen::Matrix<Scalar, 18, 1>
            {
                auto R = std::get<0>(state.getComponents()).getRotation();
                auto p = std::get<1>(state.getComponents()).getVector();
                auto v = std::get<2>(state.getComponents()).getVector();
                auto ba = std::get<3>(state.getComponents()).getVector();
                auto bomega = std::get<4>(state.getComponents()).getVector();
                Eigen::Matrix<double, 18, 1> x_x0;
                x_x0.block<3, 1>(0, 0) = Log3x3(R_predict.inverse() * R);
                x_x0.block<3, 1>(3, 0) = p - p_predict;
                x_x0.block<3, 1>(6, 0) = v - v_predict;
                x_x0.block<3, 1>(9, 0) = ba - b_a_predict;
                x_x0.block<3, 1>(12, 0) = bomega - b_omega_predict;
                x_x0.block<3, 1>(15, 0) = p - p_m_;
                // Log raw residuals
                // LOG(INFO) << "Raw residuals: \n"
                //           << x_x0.transpose();
                // Eigen::Matrix<double,15,1> tt1 = x_x0.block<15, 1>(0, 0);
                // Eigen::Matrix<double,3,1> tt2 = x_x0.block<3, 1>(15, 0);
                // LOG(INFO)<<"orires2: "<< tt1.transpose()*P_inv*tt1;
                // x_x0.block<15, 1>(0, 0) = P_inv_sqrt * x_x0.block<15, 1>(0, 0);
                // x_x0.block<3, 1>(15, 0) = C_R_inv_sqrt * (p - p_m_);
                x_x0 = Cov_total_inv_sqrt * x_x0;
                // Log scaled residuals
                // LOG(INFO) << "Scaled residuals: \n"
                //           << x_x0.transpose();
                // LOG(INFO) << "Residual norm: " << x_x0.norm();
                // LOG(INFO) << "residual norm : " << x_x0.norm();
                
                // Eigen::Matrix<double,15,1> tt3 =  x_x0.block<15, 1>(0, 0);
                // LOG(INFO)<<"resi2: "<<tt3.transpose()*tt3;
                return x_x0;
            };

            auto residual_jacobian_function = [&, this](const Statetype &state) -> Eigen::Matrix<Scalar, 18, 15>
            {
                Eigen::Matrix<double, 18, 15> jacob = Eigen::Matrix<double,18,15>::Zero();
                jacob.block<15, 15>(0, 0) = P_inv_sqrt;
                jacob.block<3, 3>(15, 3) = C_R_inv_sqrt;
                // LOG(INFO)<<"jabcod1: \n"<<jacob;
                Eigen::Matrix<double,18,15> FH = Eigen::Matrix<double,18,15>::Zero();;
                FH.block<15,15>(0,0) = Eigen::Matrix<double,15,15>::Identity();
                FH.block<3,3>(15,3) = Eigen::Matrix<double,3,3>::Identity();
                jacob = Eigen::Matrix<double,18,15>::Zero();
                jacob = Cov_total_inv_sqrt * FH;
                // LOG(INFO)<<"jabcod2: \n"<<jacob;
                // LOG(INFO)<<"jj: \n"<< (jacob.transpose() * jacob).inverse()*jacob.transpose()*Cov_total_inv_sqrt;
                // LOG(INFO)<<"jabcod: \n"<<jacob.transpose()*
                return jacob;
            };
            ConcreteFunction<Scalar, Statetype, 18> measurementFunction(measurement_function, measurement_gradient, residual_function, residual_jacobian_function);

            GaussNewtonMethod<Scalar, Statetype, 18> gradientDescent;
            // AdamMethod<Scalar, Statetype> gradientDescent;

            // Define optimization method and line search
            BacktrackingLineSearch<Scalar, Statetype, 18> lineSearch;

            // Define unconstrained optimizer
            UnconstrainedOptimizer<Scalar, Statetype, 18> unconstrainedOptimizer(gradientDescent, lineSearch);

            // Perform unconstrained optimization
            unconstrainedOptimizer.optimize(state, measurementFunction, config_);
            // auto optimized_state = unconstrainedOptimizer.getOptimizedVariables();
            // auto temp_p = std::get<1>(optimized_state.getComponents()).getVector();
            // auto temp_v = std::get<2>(optimized_state.getComponents()).getVector();
            // LOG(INFO) << "optimize p: " << temp_p(0, 0) << "," << temp_p(1, 0) << "," << temp_p(2, 0);
            // LOG(INFO) << "optimize v: " << temp_v(0, 0) << "," << temp_v(1, 0) << "," << temp_v(2, 0);
            // R_G_N_ = std::get<0>(optimized_state.getComponents()).getRotation();
            // b_a_ = std::get<3>(optimized_state.getComponents()).getVector();

            if (unconstrainedOptimizer.isSuccess())
            {
                LOG(INFO) << "Measurement update optimization succeeded.";
                
            }
            else
            {
                // LOG(INFO) << "Measurement update optimization failed: " << unconstrainedOptimizer.getReason();
            }
            // auto optimized_state = unconstrainedOptimizer.getOptimizedVariables();
            //     p_G_N_ = std::get<1>(optimized_state.getComponents()).getVector();
            //     v_G_N_ = std::get<2>(optimized_state.getComponents()).getVector();
            //     R_G_N_ = std::get<0>(optimized_state.getComponents()).getRotation();
            //     b_a_ = std::get<3>(optimized_state.getComponents()).getVector();
            //     b_omega_ = std::get<4>(optimized_state.getComponents()).getVector();

            R_G_N_ = R_inloop;
            p_G_N_ = p_inloop;
            v_G_N_ = v_inloop;
            b_a_ = b_a_inloop;
            b_omega_ = b_omega_inloop;
            LOG(INFO) << "after pose: " << "\n"
                      << p_G_N_;
            LOG(INFO) << "after velocity: \n"
                      << v_G_N_;
        }

    private:
        // state variable
        Eigen::Matrix<double, 3, 1> p_G_N_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);
        Eigen::Matrix<double, 3, 1> v_G_N_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);
        Eigen::Matrix<double, 3, 3> R_G_N_ = Eigen::Matrix<double, 3, 3>::Identity();
        Eigen::Matrix<double, 3, 1> b_a_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);
        Eigen::Matrix<double, 3, 1> b_omega_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);

        Eigen::Matrix<double, 3, 1> current_p_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);
        Eigen::Matrix<double, 3, 1> current_v_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);
        Eigen::Matrix<double, 3, 3> current_R_ = Eigen::Matrix<double, 3, 3>::Identity();
        Eigen::Matrix<double, 3, 1> current_b_a_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);
        Eigen::Matrix<double, 3, 1> current_b_omega_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);
        // other constants
        Eigen::Matrix<double, 3, 1> g_G_ = Eigen::Matrix<double, 3, 1>{0, 0, 9.8};

        // inpute
        Eigen::Matrix<double, 3, 1> omega_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);
        Eigen::Matrix<double, 3, 1> a_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);

        // covariance
        Eigen::Matrix<double, 15, 15> P_ = Eigen::Matrix<double, 15, 15>::Constant(0.0);
        Eigen::Matrix<double, 6, 6> C_Q_ = Eigen::Matrix<double, 6, 6>::Identity();
        Eigen::Matrix<double, 3, 3> C_R_ = Eigen::Matrix<double, 3, 3>::Identity();
        Eigen::Matrix<double, 15, 15> F_x_ = Eigen::Matrix<double, 15, 15>::Constant(0.0);
        Eigen::Matrix<double, 15, 6> F_w_ = Eigen::Matrix<double, 15, 6>::Constant(0.0);
        Eigen::Matrix<double, 3, 15> H_ = Eigen::Matrix<double, 3, 15>::Constant(0.0);

        // measures
        Eigen::Matrix<double, 3, 1> p_m_ = Eigen::Matrix<double, 3, 1>::Constant(0.0);

        int total_iter_num_ = 1;
        UnifiedOptimizerConfig config_;
    };

} // namespace rbl
