cmake_minimum_required(VERSION 3.4...3.18)
project(leaf_masking)

set(LEAFMASK_VERSION_INFO "1.0")

#find_package(Python COMPONENTS Interpreter Development) # WARNING this can make change target Python version!

find_package(pybind11 REQUIRED PATHS "c:/Users/Dani/Documents/Projects/pydev_venv/Lib/site-packages/pybind11/share/cmake/pybind11")


find_package(OpenCV REQUIRED PATHS "c:/Users/Dani/Documents/Cpp Projects/Libraries/opencv_410_dynamic")
#find_package(OpenCV REQUIRED PATHS "c:/Users/Dani/Documents/Cpp Projects/Libraries/opencv_410_static/x64/vc17/staticlib")
include_directories( ${OpenCV_INCLUDE_DIRS} )

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MT") # for static linking

pybind11_add_module(leaf_masking leaf_masking.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(leaf_masking PRIVATE VERSION_INFO=${LEAFMASK_VERSION_INFO})

target_link_libraries( leaf_masking PRIVATE ${OpenCV_LIBS} ) # for dynamic

install( DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}/Release/
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/..
    FILES_MATCHING PATTERN "*.pyd"
    )
