#ifndef GRADIENT_DESCENT_PARAMETERS_H
#define GRADIENT_DESCENT_PARAMETERS_H

#include <yaml-cpp/yaml.h>

struct GradientDescentParameters {
    double learning_rate;

    void loadFromYaml(const YAML::Node& node) {
        learning_rate = node["learning_rate"].as<double>();
    }
};

#endif // GRADIENT_DESCENT_PARAMETERS_H
