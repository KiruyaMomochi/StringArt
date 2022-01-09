#pragma once
#ifndef PIN_HPP
#define PIN_HPP

#define _USE_MATH_DEFINES
#include <cmath>

#include <functional>
#include <algorithm>
#include <utility>

std::pair<std::unique_ptr<int[]>, std::unique_ptr<int[]>> all_pin_cords(int center_x, int center_y, int radius, int pin_count)
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

#endif // PIN_HPP
