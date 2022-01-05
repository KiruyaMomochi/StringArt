#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <filesystem>
#include <fmt/format.h>
#include <l2_norm.cuh>
#include <image_process.hpp>
#include <string_art.hpp>
#include <pin.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <functional>
#include <algorithm>
#include <utility>
#include <random>

int main()
{
    auto file_name = std::filesystem::path{R"(C:\Users\xtyzw\Projects\StringArt\cat.png)"};
    // auto pin_count = 64;
    auto pin_count = 48;

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
    auto resize_img = crop(gray_img, width, height, gray_channels, min_width, min_width);
    width = height = min_width;
    min_width = std::min(width, 512);
    resize_img = crop(gray_img, width, height, gray_channels, min_width, min_width);
    width = height = min_width;

    auto stretched_img = stretch_hist(resize_img, width, height, gray_channels, 0);
    gray_channels = 1;

    auto cropped_img = crop_circle(stretched_img, width, height, min_width / 2, gray_channels);
    auto inverted_img = invert(cropped_img, width, height, gray_channels);
    auto [pin_x, pin_y] = all_pin_cords(width / 2, height / 2, min_width / 2, pin_count);
    plot_pin(cropped_img, width, height, pin_x, pin_y, pin_count);

    fmt::print("Preprocessing done\n");

    auto [pin_start, pin_end, string_img] = add_all_strings<int64_t>(
        pin_x.get(), pin_y.get(), pin_count, pin_count * pin_count * 10,
        inverted_img.get(), width);
    for (int i = 0; i < pin_count; i++)
        fmt::print("Pin {}: ({}, {})\n", i, pin_start[i], pin_end[i]);

    auto inverted_name = file_name.replace_filename("inverted.jpg");
    auto inverted_result = stbi_write_jpg(inverted_name.string().c_str(),
                                          width, height, 1, inverted_img.get(), 100);

    auto output_name = file_name.replace_filename("output.jpg");
    auto write_result = stbi_write_jpg(output_name.string().c_str(),
                                       width, height, 1, string_img.get(), 100);

    fmt::print("{} {} {} {}\n", width, height, gray_channels, write_result, gray_channels == 2 ? 1 : -1);

    if (!write_result)
    {
        throw std::runtime_error(fmt::format("Failed to write {}", output_name.string()));
    }
}
