project(algorithms_for_optimization)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)

include(cmake/glog.cmake)
find_package(Eigen3 3.3 REQUIRED)
find_package(YAML-CPP 0.6.2 REQUIRED)

set(${PROJECT_NAME}_LIBS
    ${YAML_CPP_LIBRARIES}
    ${GLOG_LIBRARIES}
)
include_directories(${EIGEN3_INCLUDE_DIR} 
${GLOG_INCLUDE_DIRS}
${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ex1 chap4_local_descent/examples/example1.cpp)
target_link_libraries(ex1 PUBLIC ${${PROJECT_NAME}_LIBS})