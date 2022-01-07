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
#include <cxxopts.hpp>

int main(int argc, char **argv)
{
    auto options = cxxopts::Options{"string_art", "A string art generator"};

    // clang-format off
    options.add_options()
        ("i,input", "Input image", cxxopts::value<std::string>())
        ("o,output", "Output image", cxxopts::value<std::string>()->default_value(""))
        ("p,pin-count", "Number of pins", cxxopts::value<int>()->default_value("256"))
        ("s,string-count", "Number of strings", cxxopts::value<int>()->default_value("0"))
        ("w,width", "Width of output image", cxxopts::value<int>()->default_value("512"))
        ("c,cpu", "Use CPU instead of GPU")
        ("h,help", "Print help")
        ("debug", "Print debug info");
    auto result = options.parse(argc, argv);
    // clang-format on

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (result.count("debug"))
    {
        spdlog::set_level(spdlog::level::debug);
    }

    auto file_path = result["input"].as<std::string>();
    auto file_path_fs = std::filesystem::path{file_path};
    auto pin_count = result["pin-count"].as<int>();
    auto pixel_count = result["width"].as<int>();
    auto strings_count = result["string-count"].as<int>();
    if (strings_count == 0)
        strings_count = pin_count * pin_count * 2;

    // auto file_path_fs = std::filesystem::path{R"(/home/kyaru/cs121/StringArt/input/einstein_denoised.png)"};
    // auto pin_count = 256;
    // auto pixel_count = 1024;
    // auto strings_count = pin_count * pin_count * 5;
    auto base_name = file_path_fs.stem().string();
    // auto inverted_path = file_path_fs.replace_filename(fmt::format("{}_inverted.jpg", base_name)).string();
    auto output_path = result["output"].as<std::string>();
    if (output_path.empty())
    {
        output_path = file_path_fs.replace_filename(fmt::format("{}_output_{}_{}_{}.jpg", base_name, pixel_count, pin_count, strings_count)).string();
    }

    int width, height, channels;

    auto img =
        std::unique_ptr<unsigned char[]>(stbi_load(file_path.c_str(), &width, &height, &channels, 0));

    spdlog::info("File {} loaded with {}x{}x{}", file_path, width, height, channels);
    spdlog::info("{} strings, {} pins, {} pixels", strings_count, pin_count, pixel_count);

    if (img.get() == nullptr)
    {
        throw std::runtime_error(fmt::format("Failed to load {}", file_path_fs.string()));
    }

    int gray_channels;
    auto gray_img = gray_scale(img, width, height, channels, gray_channels);
    assert(gray_channels == 1);
    spdlog::debug("Gray scale image created");

    auto min_width = std::min(width, height);
    auto resize_img = crop(gray_img, width, height, gray_channels, min_width, min_width);
    width = height = min_width;
    spdlog::debug("Image cropped");

    resize_img = resize(gray_img, width, height, gray_channels, pixel_count, pixel_count);
    width = height = pixel_count;
    spdlog::debug("Image resized to {}x{}", width, height);

    auto stretched_img = stretch_hist(resize_img, width, height, gray_channels, 0);
    assert(gray_channels == 1);
    spdlog::debug("Image stretched");

    auto cropped_img = crop_circle(stretched_img, width, height, width / 2, gray_channels);
    spdlog::debug("Image circled");

    auto inverted_img = invert(cropped_img, width, height, gray_channels);
    spdlog::debug("Image inverted");

    auto [pin_x, pin_y] = all_pin_cords(width / 2, height / 2, width / 2, pin_count);
    plot_pin(cropped_img, width, height, pin_x, pin_y, pin_count);
    spdlog::debug("Pins plotted");

    spdlog::info("Preprocessing done");

    auto add_all = result.count("cpu") ? add_all_strings<int64_t> : add_all_strings_cuda<int64_t>;
    auto [pin_start, pin_end, string_img] = add_all(
        pin_x.get(), pin_y.get(), pin_count, strings_count,
        inverted_img.get(), width);

    spdlog::info("Strings added");

    for (int i = 0; i < strings_count; i++)
    {
        if (pin_start[i] != pin_end[i])
            spdlog::debug("Pin {}: ({}, {})", i, pin_start[i], pin_end[i]);
    }

    // auto inverted_result = stbi_write_jpg(inverted_path.c_str(),
    //                                       width, height, 1, inverted_img.get(), 100);
    // spdlog::info("Inverted image saved to {}", inverted_path);

    auto output_inverted = invert(string_img, width, height, 1);
    auto write_result = stbi_write_jpg(output_path.c_str(),
                                       width, height, 1, output_inverted.get(), 100);
    spdlog::info("Output image saved to {}", output_path);

    if (!write_result)
    {
        throw std::runtime_error(fmt::format("Failed to write {}", output_path));
    }

    return 0;
}
