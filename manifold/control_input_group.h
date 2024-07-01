#ifndef CONTROL_INPUT_GROUP_H
#define CONTROL_INPUT_GROUP_H

#include <tuple>
#include "control_input_base.h"

template<typename Scalar, int Dim = Eigen::Dynamic, typename... ControlInputs>
class ControlInputGroup {
public:
    using ControlInputTuple = std::tuple<ControlInputs...>;
    static constexpr int TotalDim = (ControlInputs::getDimension() + ...);

    ControlInputGroup(const ControlInputs&... control_inputs)
        : control_inputs_(control_inputs...) {}

    ControlInputGroup(const ControlInputTuple& control_inputs)
        : control_inputs_(control_inputs) {}

    ControlInputGroup() : control_inputs_() {}

    const ControlInputTuple& getControlInputs() const {
        return control_inputs_;
    }

    Eigen::Matrix<Scalar, TotalDim, 1> getValues() const {
        Eigen::Matrix<Scalar, TotalDim, 1> values;
        int offset = 0;
        applyGetValue(values, offset);
        return values;
    }

private:
    ControlInputTuple control_inputs_;

    void applyGetValue(Eigen::Matrix<Scalar, TotalDim, 1>& values, int& offset) const {
        forEachControlInput([&](const auto& control_input) {
            constexpr int dim = control_input.getDimension();
            values.template segment<dim>(offset) = control_input.getValue();
            offset += dim;
        });
    }

    template<typename Func, typename Tuple, std::size_t... Indices>
    void forEachControlInputImpl(Func&& func, Tuple& tuple, std::index_sequence<Indices...>) const {
        (..., func(std::get<Indices>(tuple)));
    }

    template<typename Func, typename Tuple>
    void forEachControlInput(Func&& func, Tuple& tuple) const {
        forEachControlInputImpl(std::forward<Func>(func), tuple, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
    }
};

#endif // CONTROL_INPUT_GROUP_H
