cmake_minimum_required(VERSION 3.18)
project(CodecCore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Try to find Zstd (required for Zstd encoder/decoder)
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD libzstd)
endif()

if(NOT ZSTD_FOUND)
    # Try manual search
    find_path(ZSTD_INCLUDE_DIR zstd.h)
    find_library(ZSTD_LIBRARY zstd)
    if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
        set(ZSTD_FOUND TRUE)
        set(ZSTD_INCLUDE_DIRS ${ZSTD_INCLUDE_DIR})
        set(ZSTD_LIBRARIES ${ZSTD_LIBRARY})
    endif()
endif()

# Try to find nvCOMP (optional, for GPU Zstd decode)
find_path(NVCOMP_INCLUDE_DIR nvcomp.hpp
    PATHS
    /usr/local/include
    /usr/include
    $ENV{NVCOMP_ROOT}/include
)

find_library(NVCOMP_LIBRARY nvcomp
    PATHS
    /usr/local/lib
    /usr/lib
    $ENV{NVCOMP_ROOT}/lib
)

if(NVCOMP_INCLUDE_DIR AND NVCOMP_LIBRARY)
    set(NVCOMP_FOUND TRUE)
    message(STATUS "nvCOMP found: ${NVCOMP_LIBRARY}")
    add_definitions(-DNVCOMP_AVAILABLE)
else()
    set(NVCOMP_FOUND FALSE)
    message(STATUS "nvCOMP not found - Zstd will use CPU fallback")
endif()

# Source files (including rANS for compatibility)
set(SOURCES
    encoder_simple.cpp
    decoder_host.cpp
    rans.cpp
    c_api.cpp
    encoder_batched.cpp
    decoder_batched_cpu.cpp
    decoder_batched.cpp
    c_api_batched.cpp
)

# Add Zstd files if Zstd is available
if(ZSTD_FOUND)
    list(APPEND SOURCES
        encoder_zstd_v3.cpp
        decoder_zstd_v3.cpp
        c_api_zstd.cpp
    )
    message(STATUS "Zstd support enabled (using nvCOMP 3.0.6 API)")
else()
    message(WARNING "Zstd not found - Zstd encoder/decoder will not be built")
endif()

set(CUDA_SOURCES
    decoder_gpu.cu
)

# Note: decoder_batched.cu excluded due to compile errors
# All rANS batched functionality works via CPU decoder (decoder_batched_cpu.cpp)
# If you need GPU batched decode, set: cmake .. -DBUILD_GPU_BATCHED_DECODER=ON
option(BUILD_GPU_BATCHED_DECODER "Build GPU batched decoder (has compile issues)" OFF)
if(BUILD_GPU_BATCHED_DECODER)
    list(APPEND CUDA_SOURCES decoder_batched.cu)
endif()

# Build shared library
add_library(codec_core SHARED
    ${SOURCES}
    ${CUDA_SOURCES}
)

# Include directories
target_include_directories(codec_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(ZSTD_FOUND)
    target_include_directories(codec_core PUBLIC ${ZSTD_INCLUDE_DIRS})
endif()

if(NVCOMP_FOUND)
    target_include_directories(codec_core PUBLIC ${NVCOMP_INCLUDE_DIR})
endif()

# Link CUDA libraries
target_link_libraries(codec_core
    CUDA::cudart
)

# Link Zstd if available
if(ZSTD_FOUND)
    target_link_libraries(codec_core ${ZSTD_LIBRARIES})
endif()

# Link nvCOMP if available
if(NVCOMP_FOUND)
    target_link_libraries(codec_core ${NVCOMP_LIBRARY})
endif()

# CUDA architectures (5090 = Ada/Blackwell = sm_90)
set_target_properties(codec_core PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;90"
    POSITION_INDEPENDENT_CODE ON
)

# Output to build directory
set_target_properties(codec_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "Codec Core Configuration:")
message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA Architectures: 80;86;89;90")

