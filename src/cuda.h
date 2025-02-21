/*
 * This file is part of CELADRO-3D-CUDA, Copyright (C) 2024, Siavash Monfared
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


#ifndef CUDA_HPP_
#define CUDA_HPP_

// For device code compiled by nvcc, __CUDACC__ is automatically defined.
// If you want to trigger device annotations, check __CUDACC__ directly:

#ifdef __CUDACC__
    // Typical warp size on most NVIDIA GPUs
    #ifndef WarpSize
    #define WarpSize 32
    #endif

    // Typical maximum number of threads per block on most NVIDIA GPUs
    #ifndef ThreadsPerBlock
    #define ThreadsPerBlock 512
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

