#pragma once
#ifndef UTILS_CUH
#define UTILS_CUH

template <typename T>
__device__ __host__ __inline__ void swap(T& a, T& b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

#endif // UTILS_CUH
