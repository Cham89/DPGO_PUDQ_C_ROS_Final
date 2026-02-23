cmake_minimum_required(VERSION 3.0.2)
project(distributed_pgo)

## Compile as C++17 (Required for std::filesystem)
add_compile_options(-std=c++17)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
)

## Find Eigen3
find_package(Eigen3 REQUIRED)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  PgoUpdate.msg
  PgoStatus.msg
)

## Generate added messages
generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs message_runtime
  INCLUDE_DIRS include
)

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

## --- 1. Build the Robot Node ---
add_executable(robot_node 
  src/robot_node.cpp 
  src/pudq_lib.cpp 
  src/graph.cpp 
  src/optimization.cpp
)
add_dependencies(robot_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(robot_node ${catkin_LIBRARIES})

## --- 2. Build the Evaluator Tool ---
add_executable(evaluate_results 
  src/evaluate_results.cpp 
  src/pudq_lib.cpp 
  src/graph.cpp 
  src/optimization.cpp
)
target_link_libraries(evaluate_results ${catkin_LIBRARIES})
