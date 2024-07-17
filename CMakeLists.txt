project(algorithms_for_optimization)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

include(cmake/glog.cmake)
find_package(Eigen3 3.3 REQUIRED)
find_package(YAML-CPP 0.6.2 REQUIRED)

set(${PROJECT_NAME}_LIBS
    ${YAML_CPP_LIBRARIES}
    ${GLOG_LIBRARIES}
)
add_subdirectory(3rd/GeographicLib GeographicLib)

include_directories(${PROJECT_SOURCE_DIR}/3rd/GeographicLib/include/)
include_directories(${EIGEN3_INCLUDE_DIR} 
${GLOG_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR})

# add_executable(chap4ex1 chap4_local_descent/examples/example1.cpp)
# target_link_libraries(chap4ex1 PUBLIC ${${PROJECT_NAME}_LIBS})
# add_executable(chap10ex1 chap10_constraints/examples/example1.cpp)
# target_link_libraries(chap10ex1 PUBLIC ${${PROJECT_NAME}_LIBS})
# add_executable(test_state manifold_test/state_test.cpp)
# target_link_libraries(test_state PUBLIC ${${PROJECT_NAME}_LIBS})
# add_executable(test_kalman kalman_filter/ekf03_test.cpp)
# target_link_libraries(test_kalman PUBLIC ${${PROJECT_NAME}_LIBS} libGeographiccc)
add_executable(ekf03u kalman_filter/ekf03u_test.cpp)
target_link_libraries(ekf03u PUBLIC ${${PROJECT_NAME}_LIBS} libGeographiccc)
# add_executable(ekf03c kalman_filter/ekf03c_test.cpp)
# target_link_libraries(ekf03c PUBLIC ${${PROJECT_NAME}_LIBS} libGeographiccc)