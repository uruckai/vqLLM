cmake_minimum_required(VERSION 3.18)
project(CodecCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Find CUDA (for nvCOMP compatibility)
find_package(CUDAToolkit)

# Try to find Zstd (required for Zstd encoder/decoder)
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(ZSTD libzstd)
endif()

if(NOT ZSTD_FOUND)
    # Try manual search with explicit paths
    find_path(ZSTD_INCLUDE_DIR zstd.h
        PATHS
        /usr/include
        /usr/local/include
        NO_DEFAULT_PATH
    )
    find_library(ZSTD_LIBRARY
        NAMES zstd libzstd
        PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        NO_DEFAULT_PATH
    )
    
    # If not found with NO_DEFAULT_PATH, try system defaults
    if(NOT ZSTD_INCLUDE_DIR)
        find_path(ZSTD_INCLUDE_DIR zstd.h)
    endif()
    if(NOT ZSTD_LIBRARY)
        find_library(ZSTD_LIBRARY NAMES zstd libzstd)
    endif()
    
    if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
        set(ZSTD_FOUND TRUE)
        set(ZSTD_INCLUDE_DIRS ${ZSTD_INCLUDE_DIR})
        set(ZSTD_LIBRARIES ${ZSTD_LIBRARY})
        message(STATUS "Zstd found manually:")
        message(STATUS "  Library: ${ZSTD_LIBRARY}")
        message(STATUS "  Include: ${ZSTD_INCLUDE_DIR}")
    else()
        message(WARNING "Zstd not found! Include: ${ZSTD_INCLUDE_DIR}, Library: ${ZSTD_LIBRARY}")
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

# Source files (Zstd implementation only)
set(SOURCES
    c_api_zstd.cpp
)

# Add Zstd files if Zstd is available
if(ZSTD_FOUND)
    list(APPEND SOURCES
        encoder_zstd_v3.cpp
        decoder_zstd_v3.cpp
    )
    message(STATUS "Zstd support enabled (using nvCOMP 3.0.6 API)")
else()
    message(WARNING "Zstd not found - Zstd encoder/decoder will not be built")
endif()

# CUDA sources (none needed for pure Zstd implementation)
set(CUDA_SOURCES "")

# Build shared library
add_library(codec_core SHARED
    ${SOURCES}
    ${CUDA_SOURCES}
)

# Include directories
target_include_directories(codec_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(ZSTD_FOUND)
    target_include_directories(codec_core PUBLIC ${ZSTD_INCLUDE_DIRS})
    message(STATUS "Zstd include dirs: ${ZSTD_INCLUDE_DIRS}")
    message(STATUS "Zstd libraries: ${ZSTD_LIBRARIES}")
endif()

if(NVCOMP_FOUND)
    target_include_directories(codec_core PUBLIC ${NVCOMP_INCLUDE_DIR})
endif()

# Link libraries (no CUDA needed for pure Zstd implementation)
# CUDA runtime is handled by nvCOMP internally

# Link Zstd if available
if(ZSTD_FOUND)
    target_link_libraries(codec_core ${ZSTD_LIBRARIES})
endif()

# Link nvCOMP if available
if(NVCOMP_FOUND)
    target_link_libraries(codec_core ${NVCOMP_LIBRARY})
endif()

# Set library properties
set_target_properties(codec_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Output to build directory
set_target_properties(codec_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "Codec Core Configuration:")
message(STATUS "  Implementation: Zstd via nvCOMP")
message(STATUS "  Zstd Support: ${ZSTD_FOUND}")
message(STATUS "  nvCOMP Support: ${NVCOMP_FOUND}")

