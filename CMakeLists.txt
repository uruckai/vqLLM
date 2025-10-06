cmake_minimum_required(VERSION 3.18)
project(weight_codec LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(USE_CUDA "Build with CUDA support" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON "Build Python bindings" ON)

# Find packages
find_package(Threads REQUIRED)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DUSE_CUDA)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/cpp/include)

# Source files
set(WCODEC_SOURCES
    cpp/src/encoder.cpp
    cpp/src/decoder.cpp
    cpp/src/predictor.cpp
    cpp/src/rans.cpp
    cpp/src/c_api.cpp
    cpp/src/transform.cpp
    cpp/src/bitplane.cpp
    cpp/src/container.cpp
)

# CUDA sources (placeholder - will be added in Week 4)
if(USE_CUDA)
    set(WCODEC_CUDA_SOURCES
        # cuda/decoder.cu
        # cuda/transform.cu
        # cuda/rans.cu
    )
endif()

# Library target
add_library(wcodec SHARED ${WCODEC_SOURCES})
target_include_directories(wcodec PUBLIC 
    ${CMAKE_SOURCE_DIR}/cpp/include
)
target_compile_features(wcodec PUBLIC cxx_std_17)

# CUDA library (placeholder for now)
# if(USE_CUDA)
#     add_library(wcodec_cuda SHARED ${WCODEC_CUDA_SOURCES})
#     target_link_libraries(wcodec_cuda CUDA::cudart)
# endif()

# CLI tools (placeholder - will be added in Week 5)
# add_executable(wcodec-encode tools/encode.cpp)
# target_link_libraries(wcodec-encode wcodec)

# add_executable(wcodec-decode tools/decode.cpp)
# target_link_libraries(wcodec-decode wcodec)

# Python bindings (placeholder - will be added in Week 5)
# if(BUILD_PYTHON)
#     find_package(pybind11 REQUIRED)
#     pybind11_add_module(wcodec_py python/bindings.cpp)
#     target_link_libraries(wcodec_py PRIVATE wcodec)
# endif()

# Tests (placeholder - will be added in Week 2)
# if(BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(tests)
# endif()

# Installation
# install(TARGETS wcodec DESTINATION lib)
# install(DIRECTORY cpp/include/ DESTINATION include)

message(STATUS "Weight Codec Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Support: ${USE_CUDA}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Python: ${BUILD_PYTHON}")

