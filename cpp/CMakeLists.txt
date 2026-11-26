cmake_minimum_required(VERSION 3.14)
project(ssp_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(ssp INTERFACE)
target_include_directories(ssp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(kalman_cli src/kalman_cli.cpp)
target_link_libraries(kalman_cli PRIVATE ssp)

add_executable(viterbi_cli src/viterbi_cli.cpp)
target_link_libraries(viterbi_cli PRIVATE ssp)
