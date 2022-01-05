// From: https://rosettacode.org/wiki/Xiaolin_Wu%27s_line_algorithm
#pragma once
#ifndef XIAOLIN_CUH
#define XIAOLIN_CUH

#ifdef __CUDA_ARCH__
#include <nvfunctional>
#endif
#include <algorithm>
#include <utility>

#include "utils.cuh"

__device__ __host__ void WuDrawLine(float x0, float y0, float x1, float y1,
#ifdef __CUDA_ARCH__
                                    const nvstd::function<void(int x, int y, float brightess)> &plot
#else
                                    const std::function<void(int x, int y, float brightess)> &plot
#endif
)
{
#ifdef __CUDA_ARCH__
    auto ipart = [](float x) -> int
    { return int(floorf(x)); };
    auto round_f = [](float x) -> float
    { return roundf(x); };
    auto fpart_f = [](float x) -> float
    { return x - floorf(x); };
    auto rfpart_f = [=](float x) -> float
    { return 1 - fpart_f(x); };
#else
    auto ipart = [](float x) -> int
    { return int(std::floor(x)); };
    auto round_f = [](float x) -> float
    { return std::round(x); };
    auto fpart_f = [](float x) -> float
    { return x - std::floor(x); };
    auto rfpart_f = [=](float x) -> float
    { return 1 - fpart_f(x); };
#endif

    const bool steep = abs(y1 - y0) > abs(x1 - x0);
    if (steep)
    {
        swap(x0, y0);
        swap(x1, y1);
    }
    if (x0 > x1)
    {
        swap(x0, x1);
        swap(y0, y1);
    }

    const float dx = x1 - x0;
    const float dy = y1 - y0;
    const float gradient = (dx == 0) ? 1 : dy / dx;

    int xpx11;
    float intery;
    {
        const float xend = round_f(x0);
        const float yend = y0 + gradient * (xend - x0);
        const float xgap = rfpart_f(x0 + 0.5);
        xpx11 = int(xend);
        const int ypx11 = ipart(yend);
        if (steep)
        {
            plot(ypx11, xpx11, rfpart_f(yend) * xgap);
            plot(ypx11 + 1, xpx11, fpart_f(yend) * xgap);
        }
        else
        {
            plot(xpx11, ypx11, rfpart_f(yend) * xgap);
            plot(xpx11, ypx11 + 1, fpart_f(yend) * xgap);
        }
        intery = yend + gradient;
    }

    int xpx12;
    {
        const float xend = round_f(x1);
        const float yend = y1 + gradient * (xend - x1);
        const float xgap = rfpart_f(x1 + 0.5);
        xpx12 = int(xend);
        const int ypx12 = ipart(yend);
        if (steep)
        {
            plot(ypx12, xpx12, rfpart_f(yend) * xgap);
            plot(ypx12 + 1, xpx12, fpart_f(yend) * xgap);
        }
        else
        {
            plot(xpx12, ypx12, rfpart_f(yend) * xgap);
            plot(xpx12, ypx12 + 1, fpart_f(yend) * xgap);
        }
    }

    if (steep)
    {
        for (int x = xpx11 + 1; x < xpx12; x++)
        {
            plot(ipart(intery), x, rfpart_f(intery));
            plot(ipart(intery) + 1, x, fpart_f(intery));
            intery += gradient;
        }
    }
    else
    {
        for (int x = xpx11 + 1; x < xpx12; x++)
        {
            plot(x, ipart(intery), rfpart_f(intery));
            plot(x, ipart(intery) + 1, fpart_f(intery));
            intery += gradient;
        }
    }
}

#endif // XIAOLIN_CUH
