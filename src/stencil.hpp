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

