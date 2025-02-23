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
#ifndef STENCIL_HPP_
#define STENCIL_HPP_

struct stencil
{
  struct shifted_array_a
  {
    unsigned data[3];

    __host__ __device__
    unsigned& operator[](int i)
    { return data[i+1]; }

    __host__ __device__
    const unsigned& operator[](int i) const
    { return data[i+1]; }
  };

  struct shifted_array_b
  {
    shifted_array_a data[3];

    __host__ __device__
    shifted_array_a & operator[](int i)
    { return data[i+1]; }

    __host__ __device__
    const shifted_array_a & operator[](int i) const
    { return data[i+1]; }
  };
  
  shifted_array_b data[3];

  __host__ __device__
  shifted_array_b& operator[](int i)
  { return data[i+1]; }

  __host__ __device__
  const shifted_array_b& operator[](int i) const
  { return data[i+1]; }
};

#endif // STENCIL_HPP_

