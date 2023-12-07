cmake_minimum_required(VERSION 3.15)
project(test_reg)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
include_directories(include)

add_library(regress src/regress.cc)
target_link_libraries(regress Eigen3::Eigen)

add_executable (test_reg src/main.cc)
target_link_libraries(test_reg regress)


