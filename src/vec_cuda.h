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


#ifndef VEC_HPP_
#define VEC_HPP_

#include <array>
#include <iostream>
#include <cmath>   // For sqrt()

/** Simple euclidean vector
 *
 * The all-time most famous class among C++ students! This is simply an array
 * with euclidean operations built on top.
 * */
template<class T, size_t D>
struct vec
{
  /** Individual components */
  T data[D];

  __host__ __device__
  vec() = default;
  __host__ __device__
  vec(const vec& v) = default;
  __host__ __device__
  vec& operator=(const vec& v) = default;

  __host__ __device__
  vec& operator=(const T& t)
  {
    for(size_t i = 0; i < D; ++i)
      data[i] = t;
    return *this;
  }

  __host__ __device__
  vec& operator+=(const vec& v)
  {
    for(size_t i = 0; i < D; ++i)
      data[i] += v.data[i];
    return *this;
  }

__host__ __device__
vec& operator%=(const vec& v)
{
  for (size_t i = 0; i < D; ++i)
  {
    // If the divisor is zero, choose what to do:
    if (v.data[i] != 0)
      data[i] %= v.data[i];
    else
      data[i] = 0;  // or handle differently if you prefer
  }
  return *this;
}

  __host__ __device__
  vec& operator-=(const vec& v)
  {
    for(size_t i = 0; i < D; ++i)
      data[i] -= v.data[i];
    return *this;
  }

  __host__ __device__
  vec& operator*=(const T& t)
  {
    for(size_t i = 0; i < D; ++i)
      data[i] *= t;
    return *this;
  }

  __host__ __device__
  vec& operator/=(const T& t)
  {
    for(size_t i = 0; i < D; ++i)
      data[i] /= t;
    return *this;
  }

  __host__ __device__
  bool operator!=(const vec& v) const
  {
    for(size_t i = 0; i < D; ++i)
      if(data[i] != v.data[i])
        return true;
    return false;
  }

  __host__ __device__
  bool operator==(const vec& v) const
  { return not (*this != v); }

  __host__ __device__
  vec operator+(const vec& v) const
  {
    vec ret;
    for(size_t i = 0; i < D; ++i)
      ret[i] = data[i] + v.data[i];
    return ret;
  }

  __host__ __device__
  vec operator-(const vec& v) const
  {
    vec ret;
    for(size_t i = 0; i < D; ++i)
      ret[i] = data[i] - v.data[i];
    return ret;
  }

  __host__ __device__
  T operator*(const vec& v) const
  {
    T ret {0};
    for(size_t i = 0; i < D; ++i)
      ret += data[i] * v.data[i];
    return ret;
  }

  __host__ __device__
  T& operator[](size_t i)
  { return data[i]; }

  __host__ __device__
  const T& operator[](size_t i) const
  { return data[i]; }

  /** Square */
  __host__ __device__
  T sq() const { return (*this) * (*this); }
  
  /** (Currently implemented like sq(); if you intend an actual norm, consider using sqrt(sq()).) */
  __host__ __device__
  T abs() const { return (*this) * (*this); }

  /** Return unit vector */
  __host__ __device__
  vec unit_vector() const { return *this / sqrt(sq()); }

  // Arithmetic friend declarations:
  template<class U, size_t E>
  __host__ __device__ friend vec<U, E> operator+(const U&, const vec<U, E>&);
  template<class U, size_t E>
  __host__ __device__ friend vec<U, E> operator+(const vec<U, E>&, const U&);
  template<class U, size_t E>
  __host__ __device__ friend vec<U, E> operator-(const U&, const vec<U, E>&);
  template<class U, size_t E>
  __host__ __device__ friend vec<U, E> operator-(const vec<U, E>&, const U&);
  template<class U, size_t E>
  __host__ __device__ friend vec<U, E> operator*(const U&, const vec<U, E>&);
  template<class U, size_t E>
  __host__ __device__ friend vec<U, E> operator*(const vec<U, E>&, const U&);
  template<class U, size_t E>
  __host__ __device__ friend vec<U, E> operator/(const vec<U, E>&, const U&);
  // The stream output operator is host-only.
  template<class U, size_t E>
  friend std::ostream& operator<<(std::ostream&, const vec<U, E>&);

  template<class U>
  explicit operator vec<U, D>() const
  {
    vec<U, D> ret;
    for(size_t i = 0; i < D; ++i)
      ret[i] = U(data[i]);
    return ret;
  }

  // For serialization:
  using value_type = T;
  __host__ __device__
  T* begin() { return data; }
  __host__ __device__
  T* end() { return data + D; }
  
  __host__ __device__
  const T* begin() const { return data; }
  __host__ __device__
  const T* end() const { return data + D; }
};

// -----------------------------------------------------------------------------
// Non-member operator implementations

template<class T, size_t D>
__host__ __device__
vec<T, D> operator+(const vec<T, D>& v, const T& t)
{
  vec<T, D> ret;
  for(size_t i = 0; i < D; ++i)
    ret[i] = v.data[i] + t;
  return ret;
}

template<class T, size_t D>
__host__ __device__
vec<T, D> operator+(const T& t, const vec<T, D>& v)
{
  vec<T, D> ret;
  for(size_t i = 0; i < D; ++i)
    ret[i] = v.data[i] + t;
  return ret;
}

template<class T, size_t D>
__host__ __device__
vec<T, D> operator-(const vec<T, D>& v, const T& t)
{
  vec<T, D> ret;
  for(size_t i = 0; i < D; ++i)
    ret[i] = v.data[i] - t;
  return ret;
}

template<class T, size_t D>
__host__ __device__
vec<T, D> operator-(const T& t, const vec<T, D>& v)
{
  vec<T, D> ret;
  for(size_t i = 0; i < D; ++i)
    ret[i] = v.data[i] - t;
  return ret;
}

template<class T, size_t D>
__host__ __device__
vec<T, D> operator*(const vec<T, D>& v, const T& t)
{
  vec<T, D> ret;
  for(size_t i = 0; i < D; ++i)
    ret[i] = v.data[i] * t;
  return ret;
}

template<class T, size_t D>
__host__ __device__
vec<T, D> operator*(const T& t, const vec<T, D>& v)
{
  vec<T, D> ret;
  for(size_t i = 0; i < D; ++i)
    ret[i] = v.data[i] * t;
  return ret;
}

template<class T, size_t D>
__host__ __device__
vec<T, D> operator/(const vec<T, D>& v, const T& t)
{
  vec<T, D> ret;
  for(size_t i = 0; i < D; ++i)
    ret[i] = v.data[i] / t;
  return ret;
}

// Ostream output operator (host-only)
template<class T, size_t D>
std::ostream& operator<<(std::ostream& stream, const vec<T, D>& v)
{
  stream << '[';
  if(D > 0)
  {
    stream << v[0];
    for(size_t i = 1; i < D; ++i)
      stream << ", " << v[i];
  }
  return stream << ']';
}

// Modulo operator for vec<unsigned, D>
template<size_t D>
__host__ __device__
vec<unsigned, D> operator%(const vec<unsigned, D>& a, const vec<unsigned, D>& b)
{
  vec<unsigned, D> ret = a;
  for(size_t i = 0; i < D; ++i)
    ret.data[i] %= b.data[i];
  return ret;
}

#endif // VEC_HPP_

