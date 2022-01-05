#pragma once
#ifndef STRING_INDEX_CUH
#define STRING_INDEX_CUH

__device__ __host__ __inline__ void index(int m, int &start, int &end)
{
    int p = threadIdx.x + blockIdx.x * blockDim.x;
    int i = p / m;
    int j = p % m;

    if (i > (m / 2 - 1) || (m % 2 == 0 && j == 0))
    {
        return;
    }
    if (i >= j)
    {
        start = (-i - 1 + m - m % 2) % m;
        end = (-j + m - m % 2) % m;
    }
    else
    {
        start = i;
        end = j;
    }
}

__device__ __host__ __inline__ int count_strings(int strings_count)
{
    return strings_count * (strings_count - 1) / 2;
}

#endif
