/*
 * This file is part of CELADRO, Copyright (C) 2016-17, Romain Mueller
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/** Notes:
 *  - __CUDACC__ is automatically defined by nvcc when compiling device code.
 *  - _CUDA_ENABLED could be a project-wide flag if you prefer to enable CUDA 
 *    features even in host-only translation units.
 *
 *  The macro CUDA_host_device is defined to be __host__ __device__ only
 *  when compiling with nvcc, or empty otherwise.
 */

#ifndef CUDA_HPP_
#define CUDA_HPP_

// Uncomment this line if your build system sets _CUDA_ENABLED when CUDA is available
//#ifdef _CUDA_ENABLED
//#  ifdef _OPENMP
//#    error "Cuda cannot be used along with OpenMP" 
//#  endif
//#endif

// For device code compiled by nvcc, __CUDACC__ is automatically defined.
// If you want to trigger device annotations, check __CUDACC__ directly:

#ifdef __CUDACC__
    // Typical warp size on most NVIDIA GPUs
    #ifndef WarpSize
    #define WarpSize 32
    #endif

    // Typical maximum number of threads per block on most NVIDIA GPUs
    #ifndef ThreadsPerBlock
    #define ThreadsPerBlock 1024
    #endif

    // CUDA_host_device: used for functions callable from both host and device
    #define CUDA_host_device __host__ __device__
#else
    // If not compiling under nvcc, leave macros defined but empty
    #ifndef WarpSize
    #define WarpSize 1
    #endif

    #ifndef ThreadsPerBlock
    #define ThreadsPerBlock 1
    #endif

    #define CUDA_host_device
#endif // __CUDACC__

#endif // CUDA_HPP_

