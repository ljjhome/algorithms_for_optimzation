#ifndef COMMON_PARAMETERS_H
#define COMMON_PARAMETERS_H

#include <yaml-cpp/yaml.h>

struct CommonParameters {
    int max_iterations;
    double epsilon_a;
    double epsilon_r;
    double epsilon_g;

    void loadFromYaml(const YAML::Node& node) {
        max_iterations = node["max_iterations"].as<int>();
        epsilon_a = node["epsilon_a"].as<double>();
        epsilon_r = node["epsilon_r"].as<double>();
        epsilon_g = node["epsilon_g"].as<double>();
    }
};

#endif // COMMON_PARAMETERS_H
