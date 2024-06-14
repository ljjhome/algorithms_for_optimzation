#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "math.hpp"
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
namespace rbl
{
    class EKF03C
    {
    public:
        EKF03C()
        {
            config_.loadFromYaml("../config/config.yaml");
        }
        ~EKF03C() = default;
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
                // LOG(INFO) << "Y:\n"
                //           << dY;
                // LOG(INFO) << "K:\n"
                //           << K;
                dX = -(Eye15x15 - K * H_) * dY2 - K * dY;
                // LOG(INFO) << "dX:\n"
                //           << dX;
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
            static Eigen::Matrix<double, 3, 3> lastR = Eigen::Matrix<double, 3, 3>::Identity();

            using Scalar = double;
            using Statetype = State<Scalar, RotationStateComponent<Scalar>, VectorStateComponent<Scalar, 3>, VectorStateComponent<Scalar, 3>, VectorStateComponent<Scalar, 3>, VectorStateComponent<Scalar, 3>>;

            VectorStateComponent<Scalar, 3> p_component(p_G_N_);
            VectorStateComponent<Scalar, 3> v_component(v_G_N_);
            RotationStateComponent<Scalar> R_component(R_G_N_);
            VectorStateComponent<Scalar, 3> b_a_component(b_a_);
            VectorStateComponent<Scalar, 3> b_omega_component(b_omega_);
            Statetype state(R_component, p_component, v_component, b_a_component, b_omega_component);

            auto C_R_inv = C_R_.inverse();
            auto P_inv = P_pred.inverse();
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

            auto hess = [](const Statetype &state) -> Eigen::Matrix<Scalar, Statetype::TotalDim, Statetype::TotalDim>
            {
                return Eigen::Matrix<double, Statetype::TotalDim, Statetype::TotalDim>::Identity();
            };
            auto eq_constraints = [](const Statetype &state) -> Eigen::Matrix<Scalar, 1, 1>
            {
                Eigen::Matrix<Scalar, 1, 1> eq;
                eq(0, 0) = 0.0; // Example constraint: velocity norm equals 1
                return eq;
            };

            Eigen::Matrix<Scalar, 3, 3> lastRinv = lastR.transpose();
            auto ineq_constraints = [&, this](const Statetype &state) -> Eigen::Matrix<Scalar, 1, 1>
            {
                Eigen::Matrix<Scalar, 1, 1> ineq;
                Eigen::Matrix<Scalar, 3, 1> judge = lastRinv * std::get<2>(state.getComponents()).getVector();
                ineq(0, 0) = -judge(0, 0);
                return ineq;
            };

            auto eq_constraints_grad = [](const Statetype &state) -> Eigen::Matrix<Scalar, 1, 15>
            {
                Eigen::Matrix<Scalar, 1, 15> eq_grad = Eigen::Matrix<Scalar, 1, 15>::Zero();
                return eq_grad;
            };

            auto ineq_constraints_grad = [&, this](const Statetype &state) -> Eigen::Matrix<Scalar, 1, 15>
            {
                Eigen::Matrix<Scalar, 1, 15> ineq_grad = Eigen::Matrix<Scalar, 1, 15>::Zero();
                ineq_grad.block<1, 3>(0, 6) = -lastRinv.block<1, 3>(0, 0);
                return ineq_grad;
            };

            ConcreteFunction<Scalar, Statetype, 1, 1> measurementFunction(measurement_function, measurement_gradient, hess, eq_constraints, ineq_constraints, eq_constraints_grad, ineq_constraints_grad);

            AdamMethod<Scalar, Statetype> gradientDescent;

            // Define optimization method and line search
            BacktrackingLineSearch<Scalar, Statetype> lineSearch;

            // Define unconstrained optimizer
            UnconstrainedOptimizer<Scalar, Statetype> unconstrainedOptimizer(gradientDescent, lineSearch);

            // constrained mehtod
            AugmentedLagrangianMethod<Scalar, Statetype, 1, 1> constrained_method;

            constrained_method.initialize(state, measurementFunction, config_);

            // constrained optimizer
            ConstrainedOptimizer<Scalar, Statetype, 1, 1> constrained_optimizer(unconstrainedOptimizer, constrained_method);

            // Perform unconstrained optimization
            constrained_optimizer.optimize(state, measurementFunction, config_);
            auto optimized_state = constrained_optimizer.getOptimizedVariables();
            auto temp_p = std::get<1>(optimized_state.getComponents()).getVector();
            auto temp_v = std::get<2>(optimized_state.getComponents()).getVector();
            LOG(INFO) << "optimize p: " << temp_p(0, 0) << "," << temp_p(1, 0) << "," << temp_p(2, 0);
            LOG(INFO) << "optimize v: " << temp_v(0, 0) << "," << temp_v(1, 0) << "," << temp_v(2, 0);
            // R_G_N_ = std::get<0>(optimized_state.getComponents()).getRotation();
            // b_a_ = std::get<3>(optimized_state.getComponents()).getVector();

            if (constrained_optimizer.isSuccess())
            {
                LOG(INFO) << "Measurement update optimization succeeded.";
                auto optimized_state = constrained_optimizer.getOptimizedVariables();
                p_G_N_ = std::get<1>(optimized_state.getComponents()).getVector();
                v_G_N_ = std::get<2>(optimized_state.getComponents()).getVector();
                R_G_N_ = std::get<0>(optimized_state.getComponents()).getRotation();
                b_a_ = std::get<3>(optimized_state.getComponents()).getVector();
                b_omega_ = std::get<4>(optimized_state.getComponents()).getVector();
            }
            else
            {
                LOG(INFO) << "Measurement update optimization failed: " << constrained_optimizer.getReason();
            }
            p_G_N_ = std::get<1>(optimized_state.getComponents()).getVector();
            v_G_N_ = std::get<2>(optimized_state.getComponents()).getVector();
            R_G_N_ = std::get<0>(optimized_state.getComponents()).getRotation();
            b_a_ = std::get<3>(optimized_state.getComponents()).getVector();
            b_omega_ = std::get<4>(optimized_state.getComponents()).getVector();

            Eigen::AngleAxisd ana(R_G_N_);
                LOG(INFO)<<"angle: "<<ana.angle()<<", axis: "<<ana.axis().transpose();
            // R_G_N_ = R_inloop;
            // p_G_N_ = p_inloop;
            // v_G_N_ = v_inloop;
            // b_a_ = b_a_inloop;
            // b_omega_ = b_omega_inloop;
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
