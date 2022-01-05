#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <filesystem>
#include <fmt/format.h>
#include <l2_norm.cuh>
#include <image_process.hpp>
#include <string_art.cuh>
#include <string_art.hpp>
#include <string_diff.cuh>
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
    int width, height, channels;
    auto pin_count = 48;

    auto img =
        std::unique_ptr<unsigned char[]>(stbi_load(file_name.string().c_str(), &width, &height, &channels, 0));
    if (img.get() == nullptr)
    {
        throw std::runtime_error(fmt::format("Failed to load {}", file_name.string()));
    }

    int gray_channels;
    auto gray_img = gray_scale(img, width, height, channels, gray_channels);
    assert(gray_channels == 1);

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

    auto my_image = std::make_unique<unsigned char[]>(width * height);
    std::fill_n(my_image.get(), width * height, 0);

    unsigned char *device_my_image = nullptr;
    cudaMalloc(&device_my_image, width * height);

    unsigned char *device_inverted_image = nullptr;
    cudaMalloc(&device_inverted_image, width * height);
    cudaMemcpy(device_inverted_image, inverted_img.get(), width * height, cudaMemcpyHostToDevice);

    int *device_pin_x = nullptr;
    cudaMalloc(&device_pin_x, pin_count * sizeof(int));
    cudaMemcpy(device_pin_x, pin_x.get(), pin_count * sizeof(int), cudaMemcpyHostToDevice);

    int *device_pin_y = nullptr;
    cudaMalloc(&device_pin_y, pin_count * sizeof(int));
    cudaMemcpy(device_pin_y, pin_y.get(), pin_count * sizeof(int), cudaMemcpyHostToDevice);

    auto [start, end, diff] = find_add_string_cuda<int64_t>(device_pin_x, device_pin_y, pin_count,
                                                       device_my_image, device_inverted_image, width);
    // auto [start, end, diff] = find_add_string<int64_t>(pin_x.get(), pin_y.get(), pin_count,
    //                                                    my_image.get(), inverted_img.get(), width);
    cudaDeviceSynchronize();

    fmt::print("{} {} {}\n", start, end, diff);

    return 0;
}
