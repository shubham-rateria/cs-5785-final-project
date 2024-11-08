cmake_minimum_required(VERSION 3.21)

project(learn)

find_package(OpenCV REQUIRED) # first line to add

set(CMAKE_CXX_STANDARD 14)

add_executable(learn main.cpp)

target_link_libraries(learn ${OpenCV_LIBS}) # second line to add