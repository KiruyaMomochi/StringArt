#pragma once
#ifndef STRING_INDEX_CUH
#define STRING_INDEX_CUH

#include <utility>

// To use std::pair, need to enable `--expt-relaxed-constexpr`
__device__ __host__ __inline__ std::pair<int, int> string_index(
    int index, int m)
{
    int i = index / m;
    int j = index % m;
    int end, start;

    if (i > m / 2 - 1 || (m % 2 == 0 && j == 0))
    {
        return std::pair<int, int>{-1, -1};
    }
    if (i >= j)
    {
        start = (-j + m - m % 2) % m;
        end = (-i - 1 + m - m % 2) % m;
    }
    else
    {
        start = j;
        end = i;
    }

    return std::make_pair(start, end);
}

__device__ __host__ __inline__ int count_strings(int strings_count)
{
    return strings_count * strings_count / 2;
}

#endif
