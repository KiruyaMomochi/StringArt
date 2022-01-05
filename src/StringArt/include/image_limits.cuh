#pragma once
#ifndef IMAGE_LIMITS_CUH
#define IMAGE_LIMITS_CUH

template <typename T>
class image_limits
{
public:
    __device__ __host__ static constexpr T min() noexcept
    {
        return T();
    }
    __device__ __host__ static constexpr T max() noexcept
    {
        return T();
    }
};

template <>
class image_limits<unsigned char>
{
public:
    __device__ __host__ static constexpr unsigned char min() noexcept
    {
        return 0;
    }
    __device__ __host__ static constexpr unsigned char max() noexcept
    {
        return 255;
    }
};

template <>
class image_limits<float>
{
public:
    __device__ __host__ static constexpr float min() noexcept
    {
        return 0.0f;
    }
    __device__ __host__ static constexpr float max() noexcept
    {
        return 1.0f;
    }
};

#endif // IMAGE_LIMITS_CUH
