#pragma once
#ifndef STRING_ART_CUH
#define STRING_ART_CUH

#include <tuple>
#include "l2_norm.cuh"
#include "xiaolin.cuh"
#include "string_diff.cuh"
#include "string_index.cuh"
#include "string_mod.cuh"
#include "string_art.hpp"

#include "maxidx/ReduceMaxIdxOptimized.cuh"

template <typename R>
__global__ void
diff_add_string_kernel(
    const unsigned char *device_my_image, const unsigned char *device_inverted_image, const int width_height,
    const int *device_pins_x, const int *device_pins_y, const int pins_count,
    R *device_diff, const int diff_count)
{
    auto th = blockIdx.x * blockDim.x + threadIdx.x;
    auto [start, end] = string_index(th, pins_count);

    if (th > diff_count || start < 0 || end < 0)
    {
        device_diff[th] = std::numeric_limits<R>::max();
        return;
    }

    if (end >= start || start >= pins_count)
    {
        device_diff[th] = std::numeric_limits<R>::max();
        return;
    }

    auto x1 = device_pins_x[start];
    auto y1 = device_pins_y[start];
    auto x2 = device_pins_x[end];
    auto y2 = device_pins_y[end];

    device_diff[th] = diff_add_string<R>(device_my_image, device_inverted_image, width_height, x1, y1, x2, y2);
}

template <typename R>
__global__ void
diff_erase_string_kernel(
    const unsigned char *device_my_image, const unsigned char *device_inverted_image, const int *device_overflow_image, const int width_height,
    const int *device_strings_start, const int *device_strings_end, const int strings_count,
    const int *device_pins_x, const int *device_pins_y, const int pins_count,
    R *device_diff)
{
    auto th = blockIdx.x * blockDim.x + threadIdx.x;

    if (th >= strings_count)
        return;

    auto start = device_strings_start[th];
    auto end = device_strings_end[th];

    // // debug code start
    // device_diff[th] = th;
    // return;
    // // debug code end

    if (end >= start || start >= strings_count)
    {
        device_diff[th] = std::numeric_limits<R>::max();
        return;
    }

    auto x1 = device_pins_x[start];
    auto y1 = device_pins_y[start];
    auto x2 = device_pins_x[end];
    auto y2 = device_pins_y[end];

    // // debug code start
    // if (th != 1068)
    //     return;
    // device_diff[0] = th;
    // device_diff[1] = start;
    // device_diff[2] = end;
    // device_diff[3] = x1;
    // device_diff[4] = y1;
    // device_diff[5] = x2;
    // device_diff[6] = y2;
    // // debug code end

    device_diff[th] = diff_erase_string<R>(device_my_image, device_inverted_image, device_overflow_image,
                                           width_height, x1, y1, x2, y2);
}

template <typename R>
__global__ void add_string_kernel(unsigned char *device_my_image,
                                  const unsigned char *device_inverted_image,
                                  int *device_overflow_image,
                                  const int width_height,
                                  const int x1, const int y1, const int x2, const int y2,
                                  R *device_diff)
{
    *device_diff = add_string<R>(device_my_image, device_inverted_image, device_overflow_image,
                                 width_height, x1, y1, x2, y2);
}

template <typename R>
std::tuple<int, int, R> find_add_string_cuda(const int *device_pins_x, const int *device_pins_y, const int pins_count,
                                             const unsigned char *device_my_image,
                                             const unsigned char *device_inverted_image,
                                             const int width_height)
{
    auto diff_count = count_strings(pins_count);

    // for (size_t i = 0; i < diff_count; i++)
    // {
    //     auto [x, y] = string_index(i, pins_count);
    //     fmt::print("{} => ({}, {})\n", i, x, y);
    // }

    R *device_diff = nullptr;
    cudaMalloc(&device_diff, diff_count * sizeof(R));
    cudaMemset(device_diff, 0, diff_count * sizeof(R));

    auto block_size = 256;
    auto grid_size = (diff_count + block_size - 1) / block_size;

    diff_add_string_kernel<R><<<grid_size, block_size>>>(device_my_image, device_inverted_image, width_height,
                                                         device_pins_x, device_pins_y, pins_count,
                                                         device_diff, diff_count);
    auto error = cudaPeekAtLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    cudaDeviceSynchronize();

    auto diff = std::make_unique<R[]>(diff_count);
    cudaMemcpy(diff.get(), device_diff, diff_count * sizeof(R), cudaMemcpyDeviceToHost);

    

    auto best_diff = std::min_element(diff.get(), diff.get() + diff_count);
    auto best_diff_index = static_cast<int>(std::distance(diff.get(), best_diff));

    auto [best_start, best_end] = string_index(best_diff_index, pins_count);

    cudaFree(device_diff);

    return std::make_tuple(best_start, best_end, *best_diff);
}

template <typename R>
std::tuple<size_t, R> find_erase_string_cuda(const int *device_pins_x, const int *device_pins_y, const int pins_count,
                                             const int *device_strings_start, const int *device_strings_end, const int strings_count,
                                             const unsigned char *device_my_image,
                                             const unsigned char *device_inverted_image,
                                             const int *device_overflow_image,
                                             const int width_height)
{
    R *device_diff = nullptr;
    cudaMalloc(&device_diff, strings_count * sizeof(R));
    cudaMemset(device_diff, 0, strings_count * sizeof(R));

    auto block_size = 256;
    auto grid_size = (strings_count + block_size - 1) / block_size;

    diff_erase_string_kernel<R><<<grid_size, block_size>>>(device_my_image, device_inverted_image, device_overflow_image, width_height,
                                                           device_strings_start, device_strings_end, strings_count,
                                                           device_pins_x, device_pins_y, pins_count,
                                                           device_diff);
    auto error = cudaPeekAtLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    cudaDeviceSynchronize();

    R *device_max = nullptr;
    cudaMalloc(&device_max, sizeof(R));

    

    auto diff = std::make_unique<R[]>(strings_count);
    cudaMemcpy(diff.get(), device_diff, strings_count * sizeof(R), cudaMemcpyDeviceToHost);

    // for (size_t i = 0; i < strings_count; i++)
    // {
    //     fmt::print("Cuda Erase string {}, new_diff: {}\n", i, diff[i]);
    // }

    auto best_diff = std::min_element(diff.get(), diff.get() + strings_count);
    auto best_diff_index = static_cast<int>(std::distance(diff.get(), best_diff));

    cudaFree(device_diff);

    return std::make_tuple(best_diff_index, *best_diff);
}

template <typename R>
std::tuple<std::unique_ptr<int[]>, std::unique_ptr<int[]>, std::unique_ptr<unsigned char[]>>
add_all_strings_cuda(int *pins_x, int *pins_y, int pins_count, int strings_count,
                     unsigned char *inverted_image, int width_height)
{
    auto size = width_height * width_height;

    auto my_image = std::make_unique<unsigned char[]>(size);
    auto overflow_image = std::make_unique<int[]>(size);

    std::fill(my_image.get(), my_image.get() + size, 0);
    std::fill(overflow_image.get(), overflow_image.get() + size, 0);

    unsigned char *device_inverted_image = nullptr;
    cudaMalloc(&device_inverted_image, size);
    cudaMemcpy(device_inverted_image, inverted_image, size, cudaMemcpyHostToDevice);

    int *device_pin_x = nullptr;
    cudaMalloc(&device_pin_x, pins_count * sizeof(int));
    cudaMemcpy(device_pin_x, pins_x, pins_count * sizeof(int), cudaMemcpyHostToDevice);

    int *device_pin_y = nullptr;
    cudaMalloc(&device_pin_y, pins_count * sizeof(int));
    cudaMemcpy(device_pin_y, pins_y, pins_count * sizeof(int), cudaMemcpyHostToDevice);

    auto current_norm = l2_norm_square<R>(my_image.get(), inverted_image, size);
    fmt::print("l2 norm at start = {}\n", current_norm);

    auto strings_start = std::make_unique<int[]>(strings_count);
    std::fill_n(strings_start.get(), strings_count, -1);
    auto strings_end = std::make_unique<int[]>(strings_count);
    std::fill_n(strings_end.get(), strings_count, -1);

    int *device_strings_start = nullptr;
    cudaMalloc(&device_strings_start, strings_count * sizeof(int));
    cudaMemcpy(device_strings_start, strings_start.get(), strings_count * sizeof(int), cudaMemcpyHostToDevice);

    int *device_strings_end = nullptr;
    cudaMalloc(&device_strings_end, strings_count * sizeof(int));
    cudaMemcpy(device_strings_end, strings_end.get(), strings_count * sizeof(int), cudaMemcpyHostToDevice);

    unsigned char *device_my_image = nullptr;
    cudaMalloc(&device_my_image, size);

    int *device_overflow_image = nullptr;
    cudaMalloc(&device_overflow_image, size * sizeof(int));

    size_t strings_current_count = 0;

    auto no_add = false;
    auto no_remove = false;
    auto next_add = true;

    auto find_string_index = [&]()
    {
        static int strings_index = 0;
        for (int i = 0; i < strings_count; i++)
        {
            auto index = (strings_index + i) % strings_count;
            if (strings_start[index] == -1)
            {
                strings_index = index;
                return index;
            }
        }
        return -1;
    };

    auto remove_string = [&](int index) -> R
    {
        auto start = strings_start[index];
        auto end = strings_end[index];

        strings_start[index] = -1;
        strings_end[index] = -1;
        strings_current_count--;

        auto diff = erase_string<R>(my_image.get(), inverted_image, overflow_image.get(), width_height,
                                    pins_x[start], pins_y[start], pins_x[end], pins_y[end]);

        return diff;
    };

    auto insert_string = [&](int start, int end) -> R
    {
        auto index = find_string_index();
        if (index == -1)
            throw std::runtime_error("No space for new string");

        strings_start[index] = start;
        strings_end[index] = end;
        strings_current_count++;

        return add_string<R>(my_image.get(), inverted_image, overflow_image.get(), width_height,
                             pins_x[start], pins_y[start], pins_x[end], pins_y[end]);
    };

    // auto sum_insert = 0;
    // auto sum_remove = 0;
    // for (size_t i = 0; i < pins_count - 1; i++)
    // {
    //     auto insert_diff = insert_string(pins_count - 1, i);
    //     sum_insert += insert_diff;
    //     fmt::print("Insert diff = {}\n", insert_diff);
    // }
    // for (size_t i = 0; i < pins_count - 1; i++)
    // {
    //     auto remove_diff = remove_string(i);
    //     sum_remove += remove_diff;
    //     fmt::print("Remove diff = {}\n", remove_diff);
    // }
    // fmt::print("Sum insert = {}\n", sum_insert);
    // fmt::print("Sum remove = {}\n", sum_remove);
    // return std::make_tuple(std::move(strings_start), std::move(strings_end), std::move(my_image));

    while (true)
    {
        if (no_add && no_remove)
            break;

        if (next_add)
        {
            if (strings_current_count >= strings_count)
            {
                next_add = false;
                no_add = true;
                continue;
            }

            cudaMemcpy(device_my_image, my_image.get(), size, cudaMemcpyHostToDevice);
            auto [start, end, diff] = find_add_string_cuda<int64_t>(device_pin_x, device_pin_y, pins_count,
                                                                    device_my_image, device_inverted_image, width_height);

            if (start == end || diff >= 0)
            {
                fmt::print("{}: No add string found\n", strings_current_count);
                next_add = false;
                no_add = true;
                continue;
            }

            assert(diff < 0);

            // auto [host_start, host_end, host_diff] = find_add_string<int64_t>(pins_x, pins_y, pins_count,
            //                                                                   my_image.get(), inverted_image, width_height);
            // fmt::print("= {} {} {}\n", host_start, host_end, host_diff);

            auto insert_diff = insert_string(start, end);
            current_norm = current_norm + diff;
            fmt::print("+ {}: ({}, {}), new_diff: {} == insert diff: {}\n", strings_current_count, start, end, diff, insert_diff);
            assert(diff == insert_diff);

            // auto current_norm_verify = l2_norm_square<R>(my_image.get(), inverted_image, size);
            // fmt::print("l2 norm after add = {} == verify = {}\n", current_norm, current_norm_verify);
            // assert(current_norm == current_norm_verify);

            no_remove = false;
        }
        else
        {
            if (strings_current_count <= 0)
            {
                next_add = true;
                no_remove = true;
                continue;
            }

            cudaMemcpy(device_strings_start, strings_start.get(), strings_count * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(device_strings_end, strings_end.get(), strings_count * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(device_my_image, my_image.get(), size, cudaMemcpyHostToDevice);
            cudaMemcpy(device_overflow_image, overflow_image.get(), size * sizeof(int), cudaMemcpyHostToDevice);

            auto [index, diff] = find_erase_string_cuda<int64_t>(device_pin_x, device_pin_y, pins_count,
                                                                 device_strings_start, device_strings_end, strings_count,
                                                                 device_my_image, device_inverted_image, device_overflow_image, width_height);

            if (index == strings_current_count || diff >= 0)
            {
                fmt::print("{}: No remove string found\n", strings_current_count);
                next_add = true;
                no_remove = true;
                continue;
            }

            assert(diff < 0);

            // auto [host_index, host_diff] = find_erase_string<int64_t>(pins_x, pins_y, pins_count,
            //                                                       strings_start.get(), strings_end.get(), strings_count,
            //                                                       my_image.get(), inverted_image, overflow_image.get(), width_height);
            // fmt::print("= {}: {} {} {}\n", host_index, strings_start[host_index], strings_end[host_index], host_diff);

            auto string_start_pin = strings_start[index];
            auto string_end_pin = strings_end[index];

            auto remove_diff = remove_string(index);
            current_norm = current_norm + diff;
            fmt::print("- {}: ({}, {}), new_diff: {} == remove diff: {}\n", index, string_start_pin, string_end_pin, diff, remove_diff);
            assert(diff == remove_diff);

            // auto current_norm_verify = l2_norm_square<R>(my_image.get(), inverted_image, size);
            // fmt::print("l2 norm after remove = {} == verify = {}\n", current_norm, current_norm_verify);
            // assert(current_norm == current_norm_verify);

            no_add = false;
        }
    }

    return std::make_tuple(std::move(strings_start), std::move(strings_end), std::move(my_image));
}

#endif // STRING_ART_CUH
