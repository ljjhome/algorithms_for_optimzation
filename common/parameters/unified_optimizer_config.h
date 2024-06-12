#ifndef UNIFIED_OPTIMIZER_CONFIG_H
#define UNIFIED_OPTIMIZER_CONFIG_H

#include "common_parameters.h"
#include "gradient_descent_parameters.h"
#include "adam_parameters.h"
#include "augmented_lagrangian_parameters.h"
#include "penalty_method_parameters.h"
#include <yaml-cpp/yaml.h>
#include <string>

struct UnifiedOptimizerConfig {
    CommonParameters common;
    GradientDescentParameters gradient_descent;
    AdamParameters adam;
    AugmentedLagrangianParameters augmented_lagrangian;
    PenaltyMethodParameters penalty_method;

    void loadFromYaml(const std::string& filename) {
        YAML::Node config = YAML::LoadFile(filename);
        common.loadFromYaml(config["common"]);
        gradient_descent.loadFromYaml(config["optimizers"]["gradient_descent"]);
        adam.loadFromYaml(config["optimizers"]["adam"]);
        augmented_lagrangian.loadFromYaml(config["constrained_optimizers"]["augmented_lagrangian"]);
        penalty_method.loadFromYaml(config["constrained_optimizers"]["penalty_method"]);
    }
};

#endif // UNIFIED_OPTIMIZER_CONFIG_H
