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

#ifndef DERIVATIVES_HPP_
#define DERIVATIVES_HPP_

#include "stencil.hpp"
#include "cuda.h"   // Must come before using CUDA_host_device
// =============================================================================
// Derivatives

/** Symmetric finite difference derivative along the x direction */
inline double derivX(const field& f, const stencil& s)
{
  return .5*( f[s[+1][0][0]] - f[s[-1][0][0]] );
}


/** Symmetric finite difference derivative along the x direction */
CUDA_host_device
inline double derivX(double *f, const stencil& s)
{
  return .5*( f[s[+1][0][0]] - f[s[-1][0][0]] );
}

/** Symmetric finite difference derivative along the y direction */
inline double derivY(const field& f, const stencil& s)
{
  return .5*( f[s[0][+1][0]] - f[s[0][-1][0]] );  
}

/** Symmetric finite difference derivative along the y direction */
CUDA_host_device
inline double derivY(double *f, const stencil& s)
{
  return .5*( f[s[0][+1][0]] - f[s[0][-1][0]] );  
}

/** Symmetric finite difference derivative along the y direction */
inline double derivZ(const field& f, const stencil& s)
{
  return .5*( f[s[0][0][+1]] - f[s[0][0][-1]] );  
}

/** Symmetric finite difference derivative along the y direction */
CUDA_host_device
inline double derivZ(double *f, const stencil& s)
{
  return .5*( f[s[0][0][+1]] - f[s[0][0][-1]] );  
}

/** Five-point finite difference laplacian */
inline double laplacian(const field& f, const stencil& s)
{
   return f[s[+1][0][0]] + f[s[0][+1][0]] + f[s[-1][0][0]] + f[s[0][-1][0]] + f[s[0][0][+1]] + f[s[0][0][-1]] - 6.*f[s[0][0][0]];
}

/** Five-point finite difference laplacian */
__host__ __device__
inline double laplacian(double *f, const stencil& s)
{
   return f[s[+1][0][0]] + f[s[0][+1][0]] + f[s[-1][0][0]] + f[s[0][-1][0]] + f[s[0][0][+1]] + f[s[0][0][-1]] - 6.*f[s[0][0][0]];
}


#endif//DERIVATIVES_HPP_
