#pragma once
#ifndef L2_NORM_CUH
#define L2_NORM_CUH

#include "cuda_runtime.h"

template <typename R, typename T>
__device__ __host__ __inline__ R l2_norm_square(const T *x, const T *y, const size_t n)
{
    using R_signed = std::make_signed<R>::type;
    R sum = 0;
    for (int i = 0; i < n; ++i)
    {
        R_signed diff = R_signed(x[i]) - y[i];
        sum += diff * diff;
    }
    return sum;
}

template <typename R, typename T>
__device__ __host__ __inline__ R l2_norm_square_replace(const R& norm, const T &x, const T &old_y, const T &new_y)
{
    R diff = R(x) - old_y;
    R diff_new = R(x) - new_y;
    return norm + diff_new * diff_new - diff * diff;
}


// R should be signed
template <typename R, typename T>
__device__ __host__ __inline__ R l2_norm_square_diff(const T &x, const T &old_y, const T &new_y)
{
    R diff = R(x) - old_y;
    R diff_new = R(x) - new_y;
    return diff_new * diff_new - diff * diff;
}

#endif // L2_NORM_CUH
