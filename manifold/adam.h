#ifndef ADAM_METHOD_H
#define ADAM_METHOD_H

#include "optimization_method_interface.h"
#include <cmath>

template <typename Scalar, typename State>
class AdamMethod : public OptimizationMethodInterface<Scalar, State>
{
public:
    using StateType = typename OptimizationMethodInterface<Scalar, State>::StateType;
    using FunctionType = typename OptimizationMethodInterface<Scalar, State>::FunctionType;
    using GradientType = typename OptimizationMethodInterface<Scalar, State>::GradientType;

    AdamMethod(Scalar alpha = 0.01, Scalar beta1 = 0.9, Scalar beta2 = 0.999, Scalar epsilon = 1e-8)
        : alpha_(alpha), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), m_(GradientType::Zero()), v_(GradientType::Zero()), t_(0) {}

    virtual GradientType getUpdateDirection(const StateType &current_state, const FunctionType &function) const override
    {
        GradientType grad = function.gradient(current_state);
        ++t_; // Increment timestep

        // Update biased first moment estimate
        m_ = beta1_ * m_ + (1 - beta1_) * grad;

        // Update biased second raw moment estimate
        v_ = beta2_ * v_ + (1 - beta2_) * grad.array().square().matrix();

        // Compute bias-corrected first moment estimate
        GradientType m_hat = m_ / (1 - std::pow(beta1_, t_));

        // Compute bias-corrected second raw moment estimate
        GradientType v_hat = v_ / (1 - std::pow(beta2_, t_));

        // Update parameters using array operations for element-wise operations
        GradientType update_direction = -alpha_ * m_hat.array() / (v_hat.array().sqrt() + epsilon_).array();

        return update_direction;
    }
    void ResetParameters() override
    {
        alpha_ = 0.01;           
        beta1_ = 0.9;
        beta2_ = 0.999;   
        epsilon_ = 1e-8;         
        t_ = 0;          
        m_ = GradientType::Zero(); 
        v_ = GradientType::Zero(); 
    }

private:
    Scalar alpha_;           // Step size
    Scalar beta1_, beta2_;   // Exponential decay rates for moment estimates
    Scalar epsilon_;         // Small number to prevent any division by zero in the implementation
    mutable int t_;          // Timestep
    mutable GradientType m_; // First moment vector
    mutable GradientType v_; // Second moment vector
};

#endif // ADAM_METHOD_H
