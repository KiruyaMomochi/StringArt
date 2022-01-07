#pragma once
#ifndef IMAGE_PROCESS_CUH
#define IMAGE_PROCESS_CUH

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize.h>

#if __cplusplus > 202002L
#include <numbers>
#endif

#if __cplusplus > 201103L
constexpr double pi()
#else
double pi()
#endif
{
#if __cplusplus > 202002L
    return std::numbers::pi;
#elif __cplusplus > 201103L && !defined(__clang__)
    return std::atan(1.0) * 4.0;
#elif defined(M_PI)
    return M_PI;
#else
    return 3.141592653589793238462643383279502884L;
#endif
}

#define _USE_MATH_DEFINES
#include <cmath>
#include <functional>
#include <algorithm>
#include <stdexcept>

#include "image_limits.cuh"

template <typename T>
std::unique_ptr<T[]> gray_scale(const std::unique_ptr<T[]> &original, int width, int height, int channels, int &gray_channels, bool keep_alpha = false)
{
    if (channels == 1)
    {
        auto gray = std::make_unique<T[]>(width * height);
        gray_channels = 1;
        for (int i = 0; i < width * height; i++)
            gray[i] = original[i];
        return std::move(gray);
    }

    // Alpha channel
    if (keep_alpha)
        gray_channels = channels == 4 ? 2 : 1;
    else
        gray_channels = 1;

    auto new_image_size = width * height * gray_channels;
    auto gray = std::make_unique<T[]>(new_image_size);

    for (int i = 0; i < width * height; i++)
    {
        gray[i * gray_channels] = (original[i * channels] + original[i * channels + 1] + original[i * channels + 2]) / 3.0;
        if (gray_channels == 2)
        {
            gray[i * gray_channels + 1] = original[i * channels + 3];
        }
    }

    return std::move(gray);
}

template <typename T>
std::unique_ptr<T[]> crop(const std::unique_ptr<T[]> &original, int width, int height, int channels, int new_width, int new_height);

std::unique_ptr<unsigned char[]> crop(const std::unique_ptr<unsigned char[]> &original, int width, int height, int channels, int new_width, int new_height)
{
    auto new_image_size = new_width * new_height * channels;
    auto new_image = std::make_unique<unsigned char[]>(new_image_size);

    auto result = stbir_resize_uint8_generic(original.get(), width, height, 0,
                                             new_image.get(), new_width, new_height, 0,
                                             channels, -1, 0,
                                             STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT, STBIR_COLORSPACE_LINEAR,
                                             nullptr);
    if (!result)
    {
        throw std::runtime_error("Failed to resize image");
    }

    return std::move(new_image);
}

template <typename T>
std::unique_ptr<T[]> resize(const std::unique_ptr<T[]> &original, int width, int height, int channels, int new_width, int new_height);

std::unique_ptr<unsigned char[]> resize(const std::unique_ptr<unsigned char[]> &original, int width, int height, int channels, int new_width, int new_height)
{
    auto new_image_size = new_width * new_height * channels;
    auto new_image = std::make_unique<unsigned char[]>(new_image_size);

    auto result = stbir_resize_uint8(original.get(), width, height, 0,
                                     new_image.get(), new_width, new_height, 0,
                                     channels);
    if (!result)
    {
        throw std::runtime_error("Failed to resize image");
    }

    return std::move(new_image);
}

template <typename T>
std::unique_ptr<T[]> stretch_hist(const std::unique_ptr<T[]> &image, int width, int height, int channels)
{
    auto image_size = width * height * channels;
    auto new_image = std::make_unique<unsigned char[]>(image_size);
    auto channel_limit = std::make_unique<std::pair<int, int>[]>(channels);

    auto min = image_limits<T>::max();
    auto max = image_limits<T>::min();
    std::fill(channel_limit.get(), channel_limit.get() + channels, std::make_pair(max, min));

    for (int i = 0; i < width * height; i++)
    {
        for (int j = 0; j < channels; j++)
        {
            auto &[cmax, cmin] = channel_limit[j];
            if (image[i * channels + j] < cmin)
                cmin = image[i * channels + j];
            if (image[i * channels + j] > cmax)
                cmax = image[i * channels + j];
        }
    }

    for (int i = 0; i < width * height; i++)
    {
        for (int j = 0; j < channels; j++)
        {
            auto &[cmax, cmin] = channel_limit[j];
            auto value = (image[i * channels + j] - cmin) * static_cast<float>(image_limits<T>::max()) / (cmax - cmin);
            new_image[i * channels + j] = value;
        }
    }

    return std::move(new_image);
}

template <typename T>
std::unique_ptr<T[]> stretch_hist(const std::unique_ptr<T[]> &image, int width, int height, int channels, int channel)
{
    auto new_image = std::make_unique<unsigned char[]>(width * height);
    auto channel_limit = std::make_unique<std::pair<int, int>[]>(channels);

    auto cmin = std::numeric_limits<unsigned char>::max();
    auto cmax = std::numeric_limits<unsigned char>::min();

    for (int i = 0; i < width * height; i++)
    {
        if (image[i * channels + channel] < cmin)
            cmin = image[i * channels + channel];
        if (image[i * channels + channel] > cmax)
            cmax = image[i * channels + channel];
    }

    for (int i = 0; i < width * height; i++)
    {
        auto value = (image[i * channels + channel] - cmin) * static_cast<float>(image_limits<T>::max()) / (cmax - cmin);
        new_image[i * channels + channel] = value;
    }

    return std::move(new_image);
}

template <typename T>
std::unique_ptr<T[]> crop_circle(const std::unique_ptr<T[]> &image, int width, int height,
                                 int radius, int channels = 1,
                                 int center_x = -1, int center_y = -1)
{
    if (center_x == -1)
        center_x = width / 2;
    if (center_y == -1)
        center_y = height / 2;

    auto new_image_size = (2 * radius) * (2 * radius) * channels;
    auto new_image = std::make_unique<unsigned char[]>(new_image_size);
    std::fill_n(new_image.get(), new_image_size, image_limits<T>::max());

    for (int i = 0; i < 2 * radius; i++)
    {
        for (int j = 0; j < 2 * radius; j++)
        {
            auto x = center_x - radius + i;
            auto y = center_y - radius + j;
            if (x < 0 || x >= width || y < 0 || y >= height)
            {
                continue;
            }

            if ((i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius)
            {
                auto index = (i * (2 * radius) + j) * channels;
                for (int k = 0; k < channels; k++)
                {
                    new_image[index + k] = image[(x * width + y) * channels + k];
                }
            }
        }
    }

    return std::move(new_image);
}

template <typename T>
std::unique_ptr<T[]> invert(const std::unique_ptr<T[]> &image, int width, int height, int channels)
{
    auto new_image_size = width * height * channels;
    auto new_image = std::make_unique<unsigned char[]>(new_image_size);

    for (int i = 0; i < width * height; i++)
    {
        for (int j = 0; j < channels; j++)
        {
            new_image[i * channels + j] = image_limits<T>::max() - image[i * channels + j];
        }
    }

    return std::move(new_image);
}

#endif // IMAGE_PROCESSING_HPP
