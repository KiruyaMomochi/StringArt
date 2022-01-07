#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <filesystem>
#include <fmt/format.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "l2_norm.cuh"
#include "image_process.hpp"
#include "string_art.cuh"
#include "string_art.hpp"
#include "string_diff.cuh"
#include "string_mod.cuh"
#include "pin.hpp"

#include <functional>
#include <algorithm>
#include <utility>
#include <random>

#include <spdlog/spdlog.h>

int main()
{
    auto file_name = std::filesystem::path{R"(/home/kyaru/cs121/StringArt/ada.png)"};
    auto pin_count = 64;
    auto pixel_count = 128;
    auto strings_count = pin_count * pin_count * 2;

    int width, height, channels;

    auto img =
        std::unique_ptr<unsigned char[]>(stbi_load(file_name.string().c_str(), &width, &height, &channels, 0));

    spdlog::info("File {} loaded with {}x{}x{}", file_name.string(), width, height, channels);
    spdlog::info("{} strings, {} pins, {} pixels", strings_count, pin_count, pixel_count);

    if (img.get() == nullptr)
    {
        throw std::runtime_error(fmt::format("Failed to load {}", file_name.string()));
    }

    int gray_channels;
    auto gray_img = gray_scale(img, width, height, channels, gray_channels);
    assert(gray_channels == 1);
    spdlog::debug("Gray scale image created");

    auto min_width = std::min(width, height);
    auto resize_img = crop(gray_img, width, height, gray_channels, min_width, min_width);
    width = height = min_width;
    spdlog::debug("Image cropped");

    min_width = std::min(width, 1024);
    resize_img = resize(gray_img, width, height, gray_channels, min_width, min_width);
    width = height = min_width;
    spdlog::debug("Image resized to {}x{}", width, height);

    auto stretched_img = stretch_hist(resize_img, width, height, gray_channels, 0);
    assert(gray_channels == 1);
    spdlog::debug("Image stretched");

    auto cropped_img = crop_circle(resize_img, width, height, min_width / 2, gray_channels);
    spdlog::debug("Image circled");

    auto inverted_img = invert(cropped_img, width, height, gray_channels);
    spdlog::debug("Image inverted");

    auto [pin_x, pin_y] = all_pin_cords(width / 2, height / 2, min_width / 2, pin_count);
    plot_pin(cropped_img, width, height, pin_x, pin_y, pin_count);
    spdlog::debug("Pins plotted");

    spdlog::info("Preprocessing done");

    auto [pin_start, pin_end, string_img] = add_all_strings<int64_t>(
        pin_x.get(), pin_y.get(), pin_count, strings_count,
        inverted_img.get(), width);

    spdlog::info("Strings added");

    auto inverted_name = file_name.replace_filename("inverted.jpg");
    auto inverted_result = stbi_write_jpg(inverted_name.string().c_str(),
                                          width, height, 1, inverted_img.get(), 100);
    spdlog::info("Inverted image saved to {}", inverted_name.string());

    auto output_name = file_name.replace_filename("output.jpg");
    auto write_result = stbi_write_jpg(output_name.string().c_str(),
                                       width, height, 1, string_img.get(), 100);
    spdlog::info("Output image saved to {}", output_name.string());

    if (!write_result)
    {
        throw std::runtime_error(fmt::format("Failed to write {}", output_name.string()));
    }

    for (int i = 0; i < strings_count; i++)
    {
        if (pin_start[i] != pin_end[i])
            fmt::print("Pin {}: ({}, {})\n", i, pin_start[i], pin_end[i]);
    }

    return 0;
}
