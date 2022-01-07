#pragma once
#ifndef IMAGE_DRAW_CUH
#define IMAGE_DRAW_CUH

template <typename T>
class image_draw
{
public:
    __device__ __host__ static constexpr T first() noexcept
    {
        return T();
    }
    __device__ __host__ static constexpr T next() noexcept
    {
        return T();
    }
};

// TODO: currently only supports first()
template <>
class image_draw<unsigned char>
{
public:
    __device__ __host__ static constexpr unsigned char first() noexcept
    {
        return 40;
    }
    __device__ __host__ static constexpr unsigned char next() noexcept
    {
        return 40;
    }
};

#endif // IMAGE_DRAW_CUH
