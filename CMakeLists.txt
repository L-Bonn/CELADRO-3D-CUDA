cmake_minimum_required(VERSION 3.10)

# Enable CUDA as well as C++.
project(celadro LANGUAGES CXX CUDA)

# ------------------------------------------------------------------------------
# Enforce C++14 for both C++ and CUDA
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# Enable separable compilation (device linking) automatically.
# ------------------------------------------------------------------------------
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ------------------------------------------------------------------------------
# Specify the CUDA architecture to target (sm_90).
# ------------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 61)

# ------------------------------------------------------------------------------
# Default build type is Release if not specified
# ------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ------------------------------------------------------------------------------
# Compiler warnings
# ------------------------------------------------------------------------------
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# ------------------------------------------------------------------------------
# CUDA-specific flags
#   -Xcompiler -std=c++14 passes -std=c++14 to the host (GCC) compiler
#   --expt-relaxed-constexpr and --expt-extended-lambda enable newer language features
#   -ftz=false       : Disable flush-to-zero mode (handle denormals normally)
#   --prec-div=true  : Enforce precise division operations
#   --prec-sqrt=true : Enforce precise square root operations
#   --fmad=false     : Disable fused multiply-add (FMA) for exact rounding
# ------------------------------------------------------------------------------
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -arch=sm_61 \
     -Xcompiler -std=c++14 \
     --expt-relaxed-constexpr \
     --expt-extended-lambda \
     -ftz=false --prec-div=true --prec-sqrt=true --fmad=false"
)

################################################################################
# Directories
################################################################################

# If your source files are in "src", add that directory.
add_subdirectory(src)

################################################################################
# Source Files
################################################################################

file(GLOB_RECURSE cpp_sources  src/*.cpp src/*.hpp)
file(GLOB_RECURSE cuda_sources src/*.cu  src/*.cuh)
set(sources ${cpp_sources} ${cuda_sources})

################################################################################
# Define the Executable
################################################################################

add_executable(celadro ${sources})

# Suppress NVCC Warning #20012 for defaulted functions in CUDA code
target_compile_options(celadro PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20012>
)

################################################################################
# Dependencies
################################################################################

# -- Find & Link CUDA
find_package(CUDAToolkit REQUIRED)
target_link_libraries(celadro PRIVATE CUDA::cudart)

# -- Boost
find_package(Boost 1.36.0 COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
    message(STATUS "Boost include directories: ${Boost_INCLUDE_DIRS}")
    target_include_directories(celadro PUBLIC ${Boost_INCLUDE_DIRS})
    target_link_libraries(celadro PUBLIC ${Boost_LIBRARIES})
endif()

# -- OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    target_compile_options(celadro PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(celadro PUBLIC OpenMP::OpenMP_CXX)
else()
    # Fallback if OpenMP not found
    set_target_properties(celadro PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Optionally handle Hydra environment
option(HYDRA "Make linking work on hydra (as of 2017)" OFF)
if(HYDRA)
    set_target_properties(celadro PROPERTIES
        LINK_FLAGS "${CMAKE_LINK_FLAGS} -L/usr/local/shared/boost/1.64.0-gcc5.4.0/lib"
    )
endif()

