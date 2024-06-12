#ifndef ADAM_PARAMETERS_H
#define ADAM_PARAMETERS_H

#include <yaml-cpp/yaml.h>

struct AdamParameters {
    double alpha;
    double beta1;
    double beta2;
    double epsilon;

    void loadFromYaml(const YAML::Node& node) {
        alpha = node["alpha"].as<double>();
        beta1 = node["beta1"].as<double>();
        beta2 = node["beta2"].as<double>();
        epsilon = node["epsilon"].as<double>();
    }
};

#endif // ADAM_PARAMETERS_H
