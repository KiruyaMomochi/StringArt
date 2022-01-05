#pragma once
#ifndef STRING_ART_CUH
#define STRING_ART_CUH

#include <tuple>
#include "l2_norm.cuh"
#include "xiaolin.cuh"
#include "string_diff.cuh"

template <typename R>
__global__ void
add_lines_kernel(
    const unsigned char *device_my_image, const unsigned char *device_inverted_image, const int width_height,
    const int *device_pins_x, const int *device_pins_y, const int pins_count,
    R *device_diff, const int diff_count)
{
    auto th = blockIdx.x * blockDim.x + threadIdx.x;
    auto start = th / pins_count;
    auto end = th % pins_count;

    if (th > diff_count)
        return;

    if (end >= start || start >= pins_count)
        return;

    auto x1 = device_pins_x[start];
    auto y1 = device_pins_y[start];
    auto x2 = device_pins_x[end];
    auto y2 = device_pins_y[end];

    device_diff[th] = diff_add_string<R>(device_my_image, device_inverted_image, width_height, x1, y1, x2, y2);
}

template <typename R>
std::tuple<int, int, R> find_add_string_cuda(const int *device_pins_x, const int *device_pins_y, const int pins_count,
                                        const unsigned char *device_my_image,
                                        const unsigned char *device_inverted_image,
                                        const int width_height)
{
    auto diff_count = pins_count * pins_count;

    R *device_diff = nullptr;
    cudaMalloc(&device_diff, diff_count * sizeof(R));
    cudaMemset(device_diff, 0, diff_count * sizeof(R));

    auto block_size = 256;
    auto grid_size = (diff_count + block_size - 1) / block_size;

    add_lines_kernel<R><<<grid_size, block_size>>>(device_my_image, device_inverted_image, width_height,
                                                   device_pins_x, device_pins_y, pins_count,
                                                   device_diff, diff_count);
    auto error = cudaPeekAtLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    cudaDeviceSynchronize();

    auto diff = std::make_unique<R[]>(diff_count);
    cudaMemcpy(diff.get(), device_diff, diff_count * sizeof(int64_t), cudaMemcpyDeviceToHost);

    auto best_diff = std::min_element(diff.get(), diff.get() + diff_count);
    auto best_diff_index = static_cast<int>(std::distance(diff.get(), best_diff));

    auto best_start = best_diff_index / pins_count;
    auto best_end = best_diff_index % pins_count;

    cudaFree(device_diff);

    return std::make_tuple(best_start, best_end, *best_diff);
}

#endif // STRING_ART_CUH
