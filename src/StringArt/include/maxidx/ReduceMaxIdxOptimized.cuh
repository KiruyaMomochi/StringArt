#pragma once
#ifndef MAXIDX__REDUCE_MAXIDX_OPTIMIZED_CUH
#define MAXIDX__REDUCE_MAXIDX_OPTIMIZED_CUH

__device__ void atomicMax(float* const address, const float value)
{
	if (*address >= value)
	{
		return;
	}

	int* const addressAsI = (int*)address;
	int old = *addressAsI, assumed;

	do 
	{
		assumed = old;
		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(addressAsI, assumed, __float_as_int(value));
	} while (assumed != old);
}

__global__ void reduceMaxIdxOptimized(const float* __restrict__ input, const int size, float* maxOut, int* maxIdxOut)
{
    float localMax = 0.f;
    int localMaxIdx = 0;

    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float val = input[i];

        if (localMax < abs(val))
        {
            localMax = abs(val);
            localMaxIdx = i;
        }
    }

    atomicMax(maxOut, localMax);

    __syncthreads();

    if (*maxOut == localMax)
    {
        *maxIdxOut = localMaxIdx;
    }
}

#endif // MAXIDX__REDUCE_MAXIDX_OPTIMIZED_CUH
