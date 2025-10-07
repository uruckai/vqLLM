cmake_minimum_required(VERSION 3.18)
project(weight_codec LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(WCODEC_ENABLE_CUDA "Build with CUDA support" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON "Build Python bindings" ON)

# Find packages
find_package(Threads REQUIRED)

# CUDA support
if(WCODEC_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        if(CUDAToolkit_FOUND)
            add_definitions(-DWCODEC_CUDA_ENABLED)
            set(CUDA_AVAILABLE TRUE)
            message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
        else()
            set(CUDA_AVAILABLE FALSE)
            message(STATUS "CUDA requested but not found, building CPU-only")
        endif()
    else()
        set(CUDA_AVAILABLE FALSE)
        message(STATUS "CUDA compiler not found, building CPU-only")
    endif()
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA disabled by option")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/cpp/include)

# CPU source files
set(WCODEC_SOURCES
    cpp/src/encoder.cpp
    cpp/src/decoder.cpp
    cpp/src/predictor.cpp
    cpp/src/rans.cpp
    cpp/src/c_api.cpp
    cpp/src/transform.cpp
    cpp/src/bitplane.cpp
    cpp/src/container.cpp
    cpp/src/gpu_decoder.cpp
)

# CUDA sources (Week 4)
if(CUDA_AVAILABLE)
    set(WCODEC_CUDA_SOURCES
        cuda/rans_decode.cu
        cuda/predictor_reconstruct.cu
        cuda/transform.cu
    )
    set_source_files_properties(${WCODEC_CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

# Library target
if(CUDA_AVAILABLE)
    add_library(wcodec SHARED ${WCODEC_SOURCES} ${WCODEC_CUDA_SOURCES})
    target_include_directories(wcodec PUBLIC 
        ${CMAKE_SOURCE_DIR}/cpp/include
        ${CMAKE_SOURCE_DIR}/cuda
    )
    target_link_libraries(wcodec PRIVATE CUDA::cudart)
    set_target_properties(wcodec PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
else()
    add_library(wcodec SHARED ${WCODEC_SOURCES})
    target_include_directories(wcodec PUBLIC 
        ${CMAKE_SOURCE_DIR}/cpp/include
    )
endif()

target_compile_features(wcodec PUBLIC cxx_std_17)
target_link_libraries(wcodec PUBLIC Threads::Threads)

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
message(STATUS "  CUDA Support: ${CUDA_AVAILABLE}")
if(CUDA_AVAILABLE)
    message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Python: ${BUILD_PYTHON}")

