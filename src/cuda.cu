/*
 * This file is part of CELADRO-3D-CUDA, Copyright (C) 2024, Siavash Monfared
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


#include "header.hpp"
#include "cuda.h"
#include "model.hpp"
#include <iostream>
#include <stdexcept>
#include <curand_kernel.h> // Required for curandState

using namespace std;

//---------------------------------------------------------------------
// Typed malloc/free on device memory
//---------------------------------------------------------------------
/*
template<class T>
void malloc_or_free(T*& ptr, size_t len, Model::ManageMemory which)
{
    if (which == Model::ManageMemory::Allocate)
        cudaMalloc((void**)&ptr, len * sizeof(T));
    else
        cudaFree(ptr);
}
*/
//---------------------------------------------------------------------
// Typed memcpy to device memory (bidirectional)
//---------------------------------------------------------------------

template<class T, class U>
void bidirectional_memcpy(T* device, U* host, size_t len, Model::CopyMemory dir) {
    if (dir == Model::CopyMemory::HostToDevice)
        cudaMemcpy(device, static_cast<const void*>(host), len * sizeof(T), cudaMemcpyHostToDevice);
    else
        cudaMemcpy(static_cast<void*>(host), device, len * sizeof(T), cudaMemcpyDeviceToHost);
}

template<class T>
void malloc_or_free(T*& ptr, size_t len, Model::ManageMemory which) {
    if (which == Model::ManageMemory::Allocate) {
        if (cudaMalloc((void**)&ptr, len * sizeof(T)) != cudaSuccess) {
            std::cerr << "CUDA malloc failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
    } else {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
}
// -----------------------------------------------------------------------------
// cuda-related functions 
// -----------------------------------------------------------------------------

// Kernel to seed the random number generator states.
// Note: The thread id now accounts for block index.
__global__ void seed_rand(curandState *state, unsigned long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

void Model::_manage_device_memoryCellBirth(ManageMemory which)
{
    malloc_or_free(d_phi, nphases * patch_N, which);       
    malloc_or_free(d_phi_old, nphases * patch_N, which);
    malloc_or_free(d_V, nphases * patch_N, which);
    malloc_or_free(d_phi_dx, nphases * patch_N, which);
    malloc_or_free(d_phi_dy, nphases * patch_N, which);
    malloc_or_free(d_phi_dz, nphases * patch_N, which);
    malloc_or_free(d_dphi, nphases * patch_N, which);
    malloc_or_free(d_dphi_old, nphases * patch_N, which);
     
    malloc_or_free(d_com, nphases, which);
    malloc_or_free(d_polarization, nphases, which);
    malloc_or_free(d_velocity, nphases, which);
    malloc_or_free(d_patch_min, nphases, which);
    malloc_or_free(d_patch_max, nphases, which);
    malloc_or_free(d_offset, nphases, which);
    malloc_or_free(d_vol, nphases, which);
    malloc_or_free(d_Fpol, nphases, which);
    
    malloc_or_free(d_stored_gam,nphases,which);
    malloc_or_free(d_stored_omega_cc,nphases,which);
    malloc_or_free(d_stored_omega_cs,nphases,which);
    malloc_or_free(d_stored_alpha,nphases,which);
    malloc_or_free(d_stored_dpol,nphases,which);
    
    malloc_or_free(d_cSxx, nphases, which);
    malloc_or_free(d_cSxy, nphases, which);
    malloc_or_free(d_cSxz, nphases, which);
    malloc_or_free(d_cSyy, nphases, which);
    malloc_or_free(d_cSyz, nphases, which);
    malloc_or_free(d_cSzz, nphases, which);
    
    malloc_or_free(d_Fpressure, nphases, which);
    malloc_or_free(d_vorticity, nphases, which);
    malloc_or_free(d_delta_theta_pol, nphases, which);
    malloc_or_free(d_theta_pol, nphases, which);
    malloc_or_free(d_theta_pol_old, nphases, which);
    malloc_or_free(d_com_x, nphases, which);
    malloc_or_free(d_com_y, nphases, which);
    malloc_or_free(d_com_z, nphases, which);

}



void Model::_manage_device_memory(ManageMemory which)
{
    malloc_or_free(d_phi, nphases * patch_N, which);       
    malloc_or_free(d_phi_old, nphases * patch_N, which);
    malloc_or_free(d_V, nphases * patch_N, which);
    malloc_or_free(d_phi_dx, nphases * patch_N, which);
    malloc_or_free(d_phi_dy, nphases * patch_N, which);
    malloc_or_free(d_phi_dz, nphases * patch_N, which);
    malloc_or_free(d_dphi, nphases * patch_N, which);
    malloc_or_free(d_dphi_old, nphases * patch_N, which);
    
    malloc_or_free(d_sum_one, N, which);
    malloc_or_free(d_sum_two, N, which);
    malloc_or_free(d_field_press, N, which);
    malloc_or_free(d_field_polx, N, which);
    malloc_or_free(d_field_poly, N, which);
    malloc_or_free(d_field_polz, N, which);
    malloc_or_free(d_field_velx, N, which);
    malloc_or_free(d_field_vely, N, which);
    malloc_or_free(d_field_velz, N, which);
    
    malloc_or_free(d_field_sxx, N, which);
    malloc_or_free(d_field_sxy, N, which);
    malloc_or_free(d_field_sxz, N, which);
    malloc_or_free(d_field_syy, N, which);
    malloc_or_free(d_field_syz, N, which);
    malloc_or_free(d_field_szz, N, which);
    
    malloc_or_free(d_neighbors, N, which);
    malloc_or_free(d_walls, N, which);
    malloc_or_free(d_walls_dx, N, which);
    malloc_or_free(d_walls_dy, N, which);
    malloc_or_free(d_walls_dz, N, which);
    malloc_or_free(d_walls_laplace, N, which);  
    malloc_or_free(d_com, nphases, which);
    malloc_or_free(d_polarization, nphases, which);
    malloc_or_free(d_velocity, nphases, which);
    malloc_or_free(d_patch_min, nphases, which);
    malloc_or_free(d_patch_max, nphases, which);
    malloc_or_free(d_offset, nphases, which);
    malloc_or_free(d_vol, nphases, which);
    malloc_or_free(d_Fpol, nphases, which);
    
    malloc_or_free(d_stored_gam,nphases,which);
    malloc_or_free(d_stored_omega_cc,nphases,which);
    malloc_or_free(d_stored_omega_cs,nphases,which);
    malloc_or_free(d_stored_alpha,nphases,which);
    malloc_or_free(d_stored_dpol,nphases,which);
    
    malloc_or_free(d_cSxx, nphases, which);
    malloc_or_free(d_cSxy, nphases, which);
    malloc_or_free(d_cSxz, nphases, which);
    malloc_or_free(d_cSyy, nphases, which);
    malloc_or_free(d_cSyz, nphases, which);
    malloc_or_free(d_cSzz, nphases, which);
    
    malloc_or_free(d_Fpressure, nphases, which);
    malloc_or_free(d_vorticity, nphases, which);
    malloc_or_free(d_delta_theta_pol, nphases, which);
    malloc_or_free(d_theta_pol, nphases, which);
    malloc_or_free(d_theta_pol_old, nphases, which);
    malloc_or_free(d_com_x, nphases, which);
    malloc_or_free(d_com_y, nphases, which);
    malloc_or_free(d_com_z, nphases, which);
    
    // random number generation states and neighbor patch
    malloc_or_free(d_rand_states, N, which);
    malloc_or_free(d_neighbors_patch, patch_N, which);

    // Allocate center-of-mass tables
    malloc_or_free(d_com_x_table, Size[0], which);
    malloc_or_free(d_com_y_table, Size[1], which);
    malloc_or_free(d_com_z_table, Size[2], which);  // Corrected: was d_com_y_table twice.
}

void Model::_copy_device_memory(CopyMemory dir)
{
    bidirectional_memcpy(d_sum_one, &sum_one[0], N, dir);
    bidirectional_memcpy(d_sum_two, &sum_two[0], N, dir);
    bidirectional_memcpy(d_field_polx, &field_polx[0], N, dir);
    bidirectional_memcpy(d_field_poly, &field_poly[0], N, dir);
    bidirectional_memcpy(d_field_polz, &field_polz[0], N, dir);
    bidirectional_memcpy(d_field_velx, &field_velx[0], N, dir);
    bidirectional_memcpy(d_field_vely, &field_vely[0], N, dir);
    bidirectional_memcpy(d_field_velz, &field_velz[0], N, dir);
    
    bidirectional_memcpy(d_stored_gam, &stored_gam[0], nphases, dir);
    bidirectional_memcpy(d_stored_omega_cc, &stored_omega_cc[0], nphases, dir);
    bidirectional_memcpy(d_stored_omega_cs, &stored_omega_cs[0], nphases, dir);
    bidirectional_memcpy(d_stored_alpha, &stored_alpha[0], nphases, dir);
    bidirectional_memcpy(d_stored_dpol, &stored_dpol[0], nphases, dir);
    
    bidirectional_memcpy(d_field_sxx, &field_sxx[0], N, dir);
    bidirectional_memcpy(d_field_sxy, &field_sxy[0], N, dir); 
    bidirectional_memcpy(d_field_sxz, &field_sxz[0], N, dir);
    bidirectional_memcpy(d_field_syy, &field_syy[0], N, dir);
    bidirectional_memcpy(d_field_syz, &field_syz[0], N, dir);
    bidirectional_memcpy(d_field_szz, &field_szz[0], N, dir);

    bidirectional_memcpy(d_cSxx, &cSxx[0], nphases, dir);
    bidirectional_memcpy(d_cSxy, &cSxy[0], nphases, dir);
    bidirectional_memcpy(d_cSxz, &cSxz[0], nphases, dir);
    bidirectional_memcpy(d_cSyy, &cSyy[0], nphases, dir);
    bidirectional_memcpy(d_cSyz, &cSyz[0], nphases, dir);
    bidirectional_memcpy(d_cSzz, &cSzz[0], nphases, dir);
        
    bidirectional_memcpy(d_field_press, &field_press[0], N, dir);
    bidirectional_memcpy(d_neighbors, &neighbors[0], N, dir);
    bidirectional_memcpy(d_walls, &walls[0], N, dir);
    bidirectional_memcpy(d_walls_dx, &walls_dx[0], N, dir);
    bidirectional_memcpy(d_walls_dy, &walls_dy[0], N, dir);
    bidirectional_memcpy(d_walls_dz, &walls_dz[0], N, dir);
    bidirectional_memcpy(d_walls_laplace, &walls_laplace[0], N, dir);
    bidirectional_memcpy(d_com, &com[0], nphases, dir);
    bidirectional_memcpy(d_polarization, &polarization[0], nphases, dir);
    bidirectional_memcpy(d_velocity, &velocity[0], nphases, dir);
    bidirectional_memcpy(d_patch_min, &patch_min[0], nphases, dir);
    bidirectional_memcpy(d_patch_max, &patch_max[0], nphases, dir);
    bidirectional_memcpy(d_offset, &offset[0], nphases, dir);
    bidirectional_memcpy(d_vol, &vol[0], nphases, dir);
    bidirectional_memcpy(d_Fpol, &Fpol[0], nphases, dir);
    bidirectional_memcpy(d_Fpressure, &Fpressure[0], nphases, dir);
    bidirectional_memcpy(d_vorticity, &vorticity[0], nphases, dir);
    bidirectional_memcpy(d_delta_theta_pol, &delta_theta_pol[0], nphases, dir);
    bidirectional_memcpy(d_theta_pol, &theta_pol[0], nphases, dir);
    bidirectional_memcpy(d_theta_pol_old, &theta_pol_old[0], nphases, dir);    
    bidirectional_memcpy(d_com_x, &com_x[0], nphases, dir);
    bidirectional_memcpy(d_com_y, &com_y[0], nphases, dir);
    bidirectional_memcpy(d_com_z, &com_z[0], nphases, dir);
    
    bidirectional_memcpy(d_neighbors_patch, &neighbors_patch[0], patch_N, dir);

    bidirectional_memcpy(d_com_x_table, &com_x_table[0], Size[0], dir);
    bidirectional_memcpy(d_com_y_table, &com_y_table[0], Size[1], dir);
    bidirectional_memcpy(d_com_z_table, &com_z_table[0], Size[2], dir);

    for (unsigned i = 0; i < nphases; ++i)
    {
    bidirectional_memcpy(d_phi + i * patch_N, &phi[i][0], patch_N, dir);
        bidirectional_memcpy(d_phi_old + i * patch_N, &phi_old[i][0], patch_N, dir);
        bidirectional_memcpy(d_V + i * patch_N, &V[i][0], patch_N, dir); 
        bidirectional_memcpy(d_phi_dx + i * patch_N, &phi_dx[i][0], patch_N, dir);
        bidirectional_memcpy(d_phi_dy + i * patch_N, &phi_dy[i][0], patch_N, dir);
        bidirectional_memcpy(d_phi_dz + i * patch_N, &phi_dz[i][0], patch_N, dir);
        bidirectional_memcpy(d_dphi + i * patch_N, &dphi[i][0], patch_N, dir);
        bidirectional_memcpy(d_dphi_old + i * patch_N, &dphi_old[i][0], patch_N, dir);
    }
}

/*
void Model::visTMP(unsigned t){
    std::vector<field> tphi;
    tphi.resize(nphases, vector<double>(patch_N, 0.));
    for (unsigned i = 0; i < nphases; ++i)
    {
    bidirectional_memcpy(d_phi + i * patch_N, &tphi[i][0], patch_N, CopyMemory::DeviceToHost);
    }
    
    field pfVis;
    pfVis.resize(N,0.);    

    for(unsigned n=nstart; n<nphases; ++n)
    {
        //PRAGMA_OMP(omp parallel for num_threads(nthreads) if(nthreads))
        for(unsigned q=0; q<patch_N; ++q){ 
        const coord GlobCoor = GetNodePosOnDomain(n,q);   
	 // pfVis[GlobCoor[1] + Size[1]*GlobCoor[0] + Size[0]*Size[1]*GlobCoor[2]]  += tphi[n][q];
	 pfVis[GlobCoor[0] + Size[0]*GlobCoor[1] + Size[0]*Size[1]*GlobCoor[2]]  += tphi[n][q];
        }  
    }
    
    
    FILE * sortie;
    char nomfic[256];
    sprintf(nomfic, "tissue_%01u.vtk", t);
    
    sortie = fopen(nomfic, "w");
    fprintf(sortie, "# vtk DataFile Version 2.0\n");
    fprintf(sortie, "volume example\n");
    fprintf(sortie, "ASCII\n");
    fprintf(sortie, "DATASET STRUCTURED_POINTS\n");    
    fprintf(sortie, "DIMENSIONS %i %i %i\n", Size[0], Size[1], Size[2]);
    fprintf(sortie, "ASPECT_RATIO  %i %i %i\n", 1, 1, 1);
    fprintf(sortie, "ORIGIN  %i %i %i\n", 0, 0, 0);    
    fprintf(sortie, "POINT_DATA  %i\n", N);
    fprintf(sortie, "SCALARS volume_scalars double 1\n");       
    fprintf(sortie, "LOOKUP_TABLE default \n");       
    size_t ik = 0;
    for (size_t z = 0 ; z < Size[2] ; z++) {
        for (size_t y = 0 ; y < Size[1] ; y++) {
            for (size_t x = 0 ; x < Size[0] ; x++) {
                fprintf(sortie, " %g ", pfVis[ik++]);
            }
        }
        fprintf(sortie,"\n");
    }
    
    fprintf(sortie, "SCALARS id float\n");
    fprintf(sortie, "LOOKUP_TABLE default \n");       
    ik = 0;
    for (size_t z = 0 ; z < Size[2] ; z++) {
        for (size_t y = 0 ; y < Size[1] ; y++) {
            for (size_t x = 0 ; x < Size[0] ; x++) {
                fprintf(sortie, " %g ", pfVis[ik++]);
            }
        }
        fprintf(sortie,"\n");
    }

    fclose(sortie);
     
}
*/

void Model::AllocDeviceMemoryCellBirth()
{
    _manage_device_memoryCellBirth(ManageMemory::Allocate);
}

void Model::FreeDeviceMemoryCellBirth()
{
    _manage_device_memoryCellBirth(ManageMemory::Free);
}



void Model::AllocDeviceMemory()
{
    _manage_device_memory(ManageMemory::Allocate);
}

void Model::FreeDeviceMemory()
{
    _manage_device_memory(ManageMemory::Free);
}

void Model::PutToDevice()
{
    _copy_device_memory(CopyMemory::HostToDevice);
}

void Model::GetFromDevice()
{
    _copy_device_memory(CopyMemory::DeviceToHost);
}

void Model::QueryDeviceProperties()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    if (devCount > 1)
        throw error_msg("multiple Cuda devices not supported.");
    if (devCount == 0)
        throw error_msg("no cuda device found.");

    cudaDeviceProp DeviceProperties;
    cudaGetDeviceProperties(&DeviceProperties, 0);

    const int kb = 1024;
    const int gb = kb * kb * kb;
    
    cout << "... device " << DeviceProperties.name 
         << " (" << DeviceProperties.major << "." << DeviceProperties.minor << ")" << endl;
    cout << "    ... global memory:        " << DeviceProperties.totalGlobalMem / gb << "gb" << endl;
    cout << "    ... shared memory:        " << DeviceProperties.sharedMemPerBlock / kb << "kb" << endl;
    cout << "    ... constant memory:      " << DeviceProperties.totalConstMem / kb << "kb" << endl;
    cout << "    ... block registers:      " << DeviceProperties.regsPerBlock << endl;
    cout << "    ... warp size:            " << DeviceProperties.warpSize << endl;
    cout << "    ... max threads per block:    " << DeviceProperties.maxThreadsPerBlock << endl;
    cout << "    ... max block dimensions: [ " << DeviceProperties.maxThreadsDim[0]
         << ", " << DeviceProperties.maxThreadsDim[1] << ", " << DeviceProperties.maxThreadsDim[2] << " ]" << endl;
    cout << "    ... max grid dimensions:  [ " << DeviceProperties.maxGridSize[0]
         << ", " << DeviceProperties.maxGridSize[1] << ", " << DeviceProperties.maxGridSize[2] << " ]" << endl;
    
    // Check that our defined warp size matches the device's warp size.
    if (DeviceProperties.warpSize != WarpSize)
        throw error_msg("warp size does not match with device value. See src/cuda.h.");
    
    // Check that our desired ThreadsPerBlock does not exceed the device maximum.
    if (ThreadsPerBlock > DeviceProperties.maxThreadsPerBlock)
        throw error_msg("number of threads per block exceeds device capability. See src/cuda.h.");
}

void Model::InitializeCUDARandomNumbers()
{
    // Launch the random seeding kernel with an appropriate grid configuration.
    int totalThreads = N;  // Assuming d_rand_states was allocated with size N.
    int threadsPerBlock = (totalThreads < ThreadsPerBlock) ? totalThreads : ThreadsPerBlock;
    int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    seed_rand<<<blocks, threadsPerBlock>>>(d_rand_states, seed);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "seed_rand launch error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

void Model::InitializeCuda()
{
    n_total   = static_cast<int>(nphases_init * patch_N);
    n_blocks  = (n_total + ThreadsPerBlock - 1) / ThreadsPerBlock;
    n_threads = ThreadsPerBlock;
    cout << "CUDA init for " << n_total << " with " << n_blocks 
         << " blocks and " << n_threads << " threads per block " << "simulating "<<nphases_init<<" ALD"<<endl;
}

