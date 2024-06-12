#ifndef AUGMENTED_LAGRANGIAN_PARAMETERS_H
#define AUGMENTED_LAGRANGIAN_PARAMETERS_H

#include <yaml-cpp/yaml.h>

struct AugmentedLagrangianParameters {
    int k_max;
    double rho_init;
    double gamma;
    double tol;
    double penalty_parameter_init;

    void loadFromYaml(const YAML::Node& node) {
        k_max = node["k_max"].as<int>();
        rho_init = node["rho_init"].as<double>();
        gamma = node["gamma"].as<double>();
        tol = node["tol"].as<double>();
        penalty_parameter_init = node["penalty_parameter_init"].as<double>();
    }
};

#endif // AUGMENTED_LAGRANGIAN_PARAMETERS_H
