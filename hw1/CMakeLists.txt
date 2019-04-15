# MIT License

# Copyright (c) 2017 Red-Portal

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.5)
project(hw1)

# cmake modules path
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

# project source files
set(SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(SOURCE_FILES ${SOURCE_DIR}/main.cpp)

message("[ Source files for ${PROJECT_NAME} ]")
foreach(SOURCE_FILE ${SOURCE_FILES})
    message(STATUS "${SOURCE_FILE}")
endforeach()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(BUILD_FLAGS
    "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -O3 -march=native -fopenmp")
message("C++ flags = ${BUILD_FLAGS}")

find_package(OpenCV REQUIRED)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${OpenCV_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    gomp)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    COMPILE_FLAGS "${BUILD_FLAGS}"
    CXX_STANDARD 14)
