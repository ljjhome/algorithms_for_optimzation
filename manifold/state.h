#ifndef STATE_H
#define STATE_H

#include "state_component_base.h"
#include <tuple>
#include <Eigen/Dense>
#include <utility>

template<typename Scalar, typename... Components>
class State {
public:
    using StateType = std::tuple<Components...>;
    static constexpr int TotalDim = (Components::getDimension() + ...);

    State(const Components&... components) : components_(components...) {}

    State(const StateType& components) : components_(components) {}

    State() : components_() {} 

    const StateType& getComponents() const {
        return components_;
    }

    template<int Dim = TotalDim>
    State boxPlus(const Eigen::Matrix<Scalar, Dim, 1>& delta) const {
        StateType new_components = components_;
        int offset = 0;
        applyBoxPlus(new_components, delta, offset);
        return State(new_components);
    }

    template<int Dim = TotalDim>
    Eigen::Matrix<Scalar, Dim, 1> boxMinus(const State& other) const {
        Eigen::Matrix<Scalar, Dim, 1> result;
        int offset = 0;
        applyBoxMinus(result, offset, other.components_);
        return result;
    }

private:
    StateType components_;

    void applyBoxPlus(StateType& new_components, const Eigen::Matrix<Scalar, TotalDim, 1>& delta, int& offset) const {
        forEachComponent([&](auto& component) {
            component = component.boxPlus(delta.template segment<component.getDimension()>(offset));
            offset += component.getDimension();
        }, new_components);
    }

    void applyBoxMinus(Eigen::Matrix<Scalar, TotalDim, 1>& result, int& offset, const StateType& other_components) const {
        forEachComponent([&](const auto& component, const auto& other_component) {
            result.template segment<component.getDimension()>(offset) = component.boxMinus(other_component);
            offset += component.getDimension();
        }, components_, other_components);
    }

    template<typename Func, typename Tuple, std::size_t... Indices>
    void forEachComponentImpl(Func&& func, Tuple& tuple, std::index_sequence<Indices...>) const {
        (..., func(std::get<Indices>(tuple)));
    }

    template<typename Func, typename Tuple>
    void forEachComponent(Func&& func, Tuple& tuple) const {
        forEachComponentImpl(std::forward<Func>(func), tuple, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
    }

    template<typename Func, typename Tuple1, typename Tuple2, std::size_t... Indices>
    void forEachComponentImpl(Func&& func, Tuple1& tuple1, Tuple2& tuple2, std::index_sequence<Indices...>) const {
        (..., func(std::get<Indices>(tuple1), std::get<Indices>(tuple2)));
    }

    template<typename Func, typename Tuple1, typename Tuple2>
    void forEachComponent(Func&& func, Tuple1& tuple1, Tuple2& tuple2) const {
        forEachComponentImpl(std::forward<Func>(func), tuple1, tuple2, std::make_index_sequence<std::tuple_size<Tuple1>::value>{});
    }
};

#endif // STATE_H
