#pragma once
#ifndef STRING_ART_HPP
#define STRING_ART_HPP

#include <tuple>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include "l2_norm.cuh"
#include "xiaolin.cuh"
#include "string_diff.cuh"
#include "string_mod.cuh"

template <typename R>
std::tuple<int, int, R> find_add_string(const int *pins_x, const int *pins_y, const int pins_count,
                                        const unsigned char *my_image,
                                        const unsigned char *inverted_image,
                                        const int width_height)
{
    R best_diff = 0;
    auto best_start = 0;
    auto best_end = 0;

    for (size_t i = 0; i < pins_count; i++)
        for (size_t j = 0; j < i; j++)
        {
            auto x1 = pins_x[i];
            auto y1 = pins_y[i];
            auto x2 = pins_x[j];
            auto y2 = pins_y[j];

            auto norm_diff = diff_add_string<R>(my_image, inverted_image,
                                                width_height, x1, y1, x2, y2);

            if (norm_diff < best_diff)
            {
                best_start = i;
                best_end = j;
                best_diff = norm_diff;
            }
        }

    return std::make_tuple(best_start, best_end, best_diff);
}

template <typename R>
std::tuple<size_t, R> find_erase_string(int *pins_x, int *pins_y, int pins_count,
                                        int *strings_start, int *strings_end, size_t strings_count,
                                        const unsigned char *my_image,
                                        const unsigned char *inverted_image,
                                        const int *overflow_image,
                                        int width_height)
{
    R best_diff = std::numeric_limits<R>::max();
    size_t best_index = strings_count;

    for (size_t i = 0; i < strings_count; i++)
    {
        auto start = strings_start[i];
        auto end = strings_end[i];
        if (start == end)
            continue;
        assert(start != -1);
        assert(end != -1);

        auto x1 = pins_x[start];
        auto y1 = pins_y[start];
        auto x2 = pins_x[end];
        auto y2 = pins_y[end];

        auto norm_diff = diff_erase_string<R>(my_image, inverted_image,
                                              overflow_image,
                                              width_height, x1, y1, x2, y2);

        if (norm_diff < best_diff)
        {
            best_index = i;
            best_diff = norm_diff;
        }
    }

    return std::make_tuple(best_index, best_diff);
}

template <typename R>
std::tuple<std::unique_ptr<int[]>, std::unique_ptr<int[]>, std::unique_ptr<unsigned char[]>>
add_all_strings(int *pins_x, int *pins_y, int pins_count, int strings_count,
                unsigned char *inverted_image, int width_height)
{
    auto size = width_height * width_height;

    auto my_image = std::make_unique<unsigned char[]>(size);
    auto overflow_image = std::make_unique<int[]>(size);

    std::fill(my_image.get(), my_image.get() + size, 0);
    std::fill(overflow_image.get(), overflow_image.get() + size, 0);

    auto current_norm = l2_norm_square<R>(my_image.get(), inverted_image, size);
    spdlog::info("l2 norm at start = {}", current_norm);

    auto strings_start = std::make_unique<int[]>(strings_count);
    auto strings_end = std::make_unique<int[]>(strings_count);

    std::fill(strings_start.get(), strings_start.get() + strings_count, -1);
    std::fill(strings_end.get(), strings_end.get() + strings_count, -1);

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

        strings_start[strings_current_count] = start;
        strings_end[strings_current_count] = end;
        strings_current_count++;

        return add_string<R>(my_image.get(), inverted_image, overflow_image.get(), width_height,
                             pins_x[start], pins_y[start], pins_x[end], pins_y[end]);
    };

    auto iteration_count = 0;
    auto sw = spdlog::stopwatch{};

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

            auto [start, end, diff] = find_add_string<R>(
                pins_x, pins_y, pins_count,
                my_image.get(), inverted_image, width_height);

            if (start == end || diff >= 0)
            {
                spdlog::debug("{}: No add string found", strings_current_count);
                next_add = false;
                no_add = true;
                continue;
            }

            assert(diff < 0);

            auto insert_diff = insert_string(start, end);
            current_norm = current_norm + diff;
            spdlog::debug("+ {}: ({}, {}), new_diff: {} == insert diff: {}", strings_current_count, start, end, diff, insert_diff);
            assert(diff == insert_diff);

            // auto current_norm_verify = l2_norm_square<R>(my_image.get(), inverted_image, size);
            // spdlog::debug("l2 norm after add = {} == verify = {}", current_norm, current_norm_verify);

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

            auto [index, diff] = find_erase_string<R>(
                pins_x, pins_y, pins_count,
                strings_start.get(), strings_end.get(), strings_count,
                my_image.get(), inverted_image, overflow_image.get(), width_height);

            if (index == strings_current_count || diff >= 0)
            {
                spdlog::debug("{}: No remove string found", strings_current_count);
                next_add = true;
                no_remove = true;
                continue;
            }

            assert(diff < 0);

            auto string_start_pin = strings_start[index];
            auto string_end_pin = strings_end[index];

            auto remove_diff = remove_string(index);
            current_norm = current_norm + diff;
            spdlog::debug("- {}: ({}, {}), new_diff: {} == remove diff: {}", index, string_start_pin, string_end_pin, diff, remove_diff);
            assert(diff == remove_diff);

            // auto current_norm_verify = l2_norm_square<R>(my_image.get(), inverted_image, size);
            // spdlog::debug("l2 norm after remove = {} == verify = {}", current_norm, current_norm_verify);

            no_add = false;
        }

        iteration_count++;
    }

    spdlog::info("l2 norm after iteration = {}", current_norm);
    spdlog::info("{}s, {} iterations", sw, iteration_count);

    return std::make_tuple(std::move(strings_start), std::move(strings_end), std::move(my_image));
}

#endif
