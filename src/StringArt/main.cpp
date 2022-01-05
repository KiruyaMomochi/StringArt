#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <filesystem>
#include <fmt/format.h>
#include "l2_norm.cuh"
#include "xiaolin.hpp"
#include "image_process.hpp"
#include "string_art.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <functional>
#include <algorithm>
#include <utility>
#include <random>

std::pair<std::unique_ptr<int[]>, std::unique_ptr<int[]>> find_pin_cords(int center_x, int center_y, int radius, int pin_count)
{
    auto x = std::make_unique<int[]>(pin_count);
    auto y = std::make_unique<int[]>(pin_count);
    auto angle_step = 2 * M_PI / pin_count;
    auto angle = 0.0;
    for (int i = 0; i < pin_count; i++)
    {
        x[i] = std::round(center_x + radius * std::cos(angle));
        y[i] = std::round(center_y + radius * std::sin(angle));
        angle += angle_step;
    }

    return std::make_pair(std::move(x), std::move(y));
}

template <typename T>
void plot_pin(
    std::unique_ptr<T[]> &image, int width, int height,
    const std::unique_ptr<int[]> &pin_x, const std::unique_ptr<int[]> &pin_y, int pin_count)
{
    for (int i = 0; i < pin_count; i++)
    {
        for (int j = pin_x[i] - 1; j <= pin_x[i] + 1; j++)
        {
            for (int k = pin_y[i] - 1; k <= pin_y[i] + 1; k++)
            {
                if (j < 0 || j >= width || k < 0 || k >= height)
                    continue;
                image[j * width + k] = image_limits<T>::max() - image[j * width + k];
            }
        }
    }
}

int main()
{
    auto file_name = std::filesystem::path{R"(C:\Users\xtyzw\Projects\StringArt\cat.png)"};
    auto pin_count = 64;

    int width, height, channels;

    auto img =
        std::unique_ptr<unsigned char[]>(stbi_load(file_name.string().c_str(), &width, &height, &channels, 0));

    if (img.get() == nullptr)
    {
        throw std::runtime_error(fmt::format("Failed to load {}", file_name.string()));
    }

    int gray_channels;
    auto gray_img = gray_scale(img, width, height, channels, gray_channels);

    auto min_width = std::min(width, height);
    auto resize_img = resize(gray_img, width, height, gray_channels, min_width, min_width);
    width = height = min_width;

    auto stretched_img = stretch_hist(resize_img, width, height, gray_channels, 0);
    gray_channels = 1;

    auto cropped_img = crop_circle(stretched_img, width, height, min_width / 2, gray_channels);
    auto inverted_img = invert(cropped_img, width, height, gray_channels);
    auto [pin_x, pin_y] = find_pin_cords(width / 2, height / 2, min_width / 2, pin_count);
    plot_pin(cropped_img, width, height, pin_x, pin_y, pin_count);


    auto [pin_start, pin_end, string_img] = add_all_strings<int64_t>(
        pin_x.get(), pin_y.get(), pin_count, pin_count * pin_count,
        inverted_img.get(), width);
    for (int i = 0; i < pin_count * pin_count; i++)
        fmt::print("Pin {}: ({}, {})\n", i, pin_start[i], pin_end[i]);

    auto output_name = file_name.replace_filename("output.jpg");
    auto write_result = stbi_write_jpg(output_name.string().c_str(),
                                       width, height, 1, string_img.get(), 100);

    fmt::print("{} {} {} {}\n", width, height, gray_channels, write_result, gray_channels == 2 ? 1 : -1);

    if (!write_result)
    {
        throw std::runtime_error(fmt::format("Failed to write {}", output_name.string()));
    }
}
