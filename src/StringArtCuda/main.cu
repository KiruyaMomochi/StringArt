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
#include "string_mod.hpp"
#include "pin.hpp"

#include <functional>
#include <algorithm>
#include <utility>
#include <random>

int main()
{
    auto file_name = std::filesystem::path{R"(C:\Users\xtyzw\Projects\StringArt\cat.png)"};
    int width, height, channels;
    auto pin_count = 128;

    auto img =
        std::unique_ptr<unsigned char[]>(stbi_load(file_name.string().c_str(), &width, &height, &channels, 0));
    if (img.get() == nullptr)
    {
        throw std::runtime_error(fmt::format("Failed to load {}", file_name.string()));
    }

    int gray_channels;
    auto gray_img = gray_scale(img, width, height, channels, gray_channels);
    assert(gray_channels == 1);

    auto inverted_name = file_name.replace_filename("inverted.jpg");
    auto inverted_result = stbi_write_jpg(inverted_name.string().c_str(),
                                          width, height, 1, gray_img.get(), 100);

    auto min_width = std::min(width, height);
    auto resize_img = crop(gray_img, width, height, gray_channels, min_width, min_width);
    width = height = min_width;
    min_width = std::min(width, 512);
    resize_img = crop(gray_img, width, height, gray_channels, min_width, min_width);
    width = height = min_width;

    auto stretched_img = stretch_hist(resize_img, width, height, gray_channels, 0);
    assert(gray_channels == 1);

    auto cropped_img = crop_circle(resize_img, width, height, min_width / 2, gray_channels);
    auto inverted_img = invert(cropped_img, width, height, gray_channels);
    auto [pin_x, pin_y] = all_pin_cords(width / 2, height / 2, min_width / 2, pin_count);
    plot_pin(cropped_img, width, height, pin_x, pin_y, pin_count);

    fmt::print("Preprocessing done\n");

    auto my_image = std::make_unique<unsigned char[]>(width * height);
    std::fill_n(my_image.get(), width * height, 0);

    auto overflow_image = std::make_unique<int[]>(width * height);
    std::fill_n(overflow_image.get(), width * height, 0);

    auto strings_count = pin_count * pin_count * 10;

    auto [pin_start, pin_end, string_img] = add_all_strings_cuda<int64_t>(
        pin_x.get(), pin_y.get(), pin_count, strings_count,
        inverted_img.get(), width);

    auto output_name = file_name.replace_filename("output.jpg");
    auto write_result = stbi_write_jpg(output_name.string().c_str(),
                                       width, height, 1, string_img.get(), 100);

    fmt::print("{} {} {} {}\n", width, height, gray_channels, write_result, gray_channels == 2 ? 1 : -1);

    if (!write_result)
    {
        throw std::runtime_error(fmt::format("Failed to write {}", output_name.string()));
    }
    
    for (int i = 0; i < strings_count; i++)
    {
        if (pin_start[i] != pin_end[i])
        fmt::print("Pin {}: ({}, {})\n", i, pin_start[i], pin_end[i]);
    }

    // while (strings_current_count < strings_count)
    // {

    //     // auto [start, end, diff] = find_add_string<int64_t>(pin_x.get(), pin_y.get(), pin_count,
    //     //                                                                   my_image.get(), inverted_img.get(),
    //     //                                                                   width);
    //     // fmt::print("{} {} {}\n", host_start, host_end, host_diff);
    //     cudaDeviceSynchronize();
    //     fmt::print("+ {} {} {}\n", start, end, diff);

    //     if (diff == 0)
    //         break;

    //     strings_start[strings_current_count] = start;
    //     strings_end[strings_current_count] = end;
    //     strings_current_count++;

    //     add_string<int64_t>(my_image.get(), inverted_img.get(), overflow_image.get(), width,
    //                         pin_x[start], pin_y[start], pin_x[end], pin_y[end]);
    // }

    // for (size_t i = 0; i < strings_current_count; i++)
    // {
    //     fmt::print("String {} = {} - {}\n", i, strings_start[i], strings_end[i]);
    // }

    // {
    //     cudaMemcpy(device_strings_start, strings_start.get(), strings_count * sizeof(int), cudaMemcpyHostToDevice);
    //     cudaMemcpy(device_strings_end, strings_end.get(), strings_count * sizeof(int), cudaMemcpyHostToDevice);
    //     cudaMemcpy(device_my_image, my_image.get(), width * height, cudaMemcpyHostToDevice);
    //     cudaMemcpy(device_overflow_image, overflow_image.get(), width * height, cudaMemcpyHostToDevice);

    //     auto [index, diff] = find_erase_string_cuda<int64_t>(device_pin_x, device_pin_y, pin_count,
    //                                                          device_strings_start, device_strings_end, strings_count,
    //                                                          device_my_image, device_inverted_image, device_overflow_image, width);
    //     auto [host_index, host_diff] = find_erase_string<int64_t>(pin_x.get(), pin_y.get(), pin_count,
    //                                                               strings_start.get(), strings_end.get(), strings_count,
    //                                                               my_image.get(), inverted_img.get(), overflow_image.get(), width);
    //     cudaDeviceSynchronize();
    //     fmt::print("- {}: {} {} {}\n", index, strings_start[index], strings_end[index], diff);
    //     fmt::print("= {}: {} {} {}\n", host_index, strings_start[host_index], strings_end[host_index], host_diff);
    // }

    return 0;
}
