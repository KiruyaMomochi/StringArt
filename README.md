# StringArt

## Usage

Using `StringArtCuda --help` to get usage information. A simple example is:
```
StringArtCuda --input input/einstein_denoised.png --pin-count 64 --width 256 --cpu
```

## Build

To compile this code, both C++ and CUDA compilers, with support for C++17, are required. Latest GNU G++, Clang++ and MSVC are tested.

Because we need to define `CMAKE_CUDA_ARCHITECTURES`, we also need a cmake with version 3.18 or higher.

Since we need to connect GitHub to download some code, you should make sure that your network environment is clean. For users in Mainland China , a proxy is recommended.

After all these requirements are met, you can run the following command to compile the code:

```bash
# Change `CMAKE_CUDA_ARCHITECTURES` to match your GPU
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=60
```
