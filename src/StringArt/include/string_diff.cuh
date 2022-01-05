#pragma once
#ifndef STRING_DIFF_CUH
#define STRING_DIFF_CUH

#include "image_draw.cuh"
#include "xiaolin.cuh"

template <typename R>
__device__ __host__
    R
    diff_add_string(const unsigned char *my_image,
                    const unsigned char *inverted_image,
                    const int &width_height, const int &x1, const int &y1, const int &x2, const int &y2)
{
    R norm_diff = 0;
    WuDrawLine(x1, y1, x2, y2, [&](int x, int y, float brightness)
               {
                   if (x >= width_height || y >= width_height || x < 0 || y < 0)
                       return;

                   auto index = x * width_height + y;

                   auto old_value = my_image[index];
                   auto old_value_cast = my_image[index];

                   auto draw_value = image_draw<unsigned char>::first();
                   if (old_value != 0)
                       draw_value = image_draw<unsigned char>::first();

#ifdef __CUDA_ARCH__
                   auto new_value = old_value + ceilf(brightness * draw_value);
#else
                   auto new_value = old_value + std::ceil(brightness * draw_value);
#endif

                   auto new_value_cast = static_cast<unsigned char>(new_value);
                   if (new_value > image_limits<unsigned char>::max())
                       new_value_cast = image_limits<unsigned char>::max();
                   
                   norm_diff += l2_norm_square_diff<R>(inverted_image[index],
                                                      old_value_cast,
                                                      new_value_cast); });
    return norm_diff;
}

template <typename R>
__device__ __host__
    R
    diff_erase_string(const unsigned char *my_image,
                      const unsigned char *inverted_image,
                      const int *overflow_image,
                      const int &width_height, const int &x1, const int &y1, const int &x2, const int &y2)
{
    R norm_diff = 0;
    WuDrawLine(x1, y1, x2, y2, [&](int x, int y, float brightness)
               {
                   if (x >= width_height || y >= width_height || x < 0 || y < 0)
                       return;

                   auto index = x * width_height + y;

                   auto old_value = my_image[index] + overflow_image[index];
                   auto old_value_cast = my_image[index];

                   auto draw_value = image_draw<unsigned char>::first();
                   if (old_value < image_draw<unsigned char>::first())
                       draw_value = image_draw<unsigned char>::first();

#ifdef __CUDA_ARCH__
                   auto new_value = old_value - ceilf(brightness * draw_value);
#else                
                   auto new_value = old_value - std::ceil(brightness * draw_value);
#endif
                   
                   auto new_value_cast = static_cast<unsigned char>(new_value);
                   if (new_value > image_limits<unsigned char>::max())
                       new_value_cast = image_limits<unsigned char>::max();

                   norm_diff += l2_norm_square_diff<R>(inverted_image[index],
                                                      old_value_cast,
                                                      new_value_cast); });
    return norm_diff;
}

#endif // STRING_DIFF_CUH
