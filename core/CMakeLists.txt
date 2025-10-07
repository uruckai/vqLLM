cmake_minimum_required(VERSION 3.18)
project(CodecCore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Source files
set(SOURCES
    encoder.cpp
    decoder_host.cpp
    c_api.cpp
)

set(CUDA_SOURCES
    decoder_gpu.cu
)

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

# Link CUDA libraries
target_link_libraries(codec_core
    CUDA::cudart
)

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

