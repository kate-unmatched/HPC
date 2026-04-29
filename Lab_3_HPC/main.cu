#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "EasyBMP.h"


// CUDA error checking
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string("CUDA error: ") +                \
                                     cudaGetErrorString(err) +                    \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                         \
    } while (0)


// Texture object based access
__global__ void medianFilterKernel(cudaTextureObject_t texObj,
    unsigned char* output,
    int width,
    int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    unsigned char window[9];
    int idx = 0;

    // 3x3 neighborhood
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            // Texture is configured with clamp address mode,
            // so out-of-bounds coordinates are clamped to nearest valid pixel.
            unsigned char window[9];
            int idx = 0;

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    window[idx++] = tex2D<unsigned char>(texObj, x + dx, y + dy);
                }
            }
        }
    }

    // Oops-proof correction:
    // tex2D for neighboring pixels must use shifted coordinates
    idx = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            window[idx++] = tex2D<unsigned char>(texObj, x + dx, y + dy);
        }
    }

    // Bubble sort for 9 elements
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8 - i; ++j) {
            if (window[j] > window[j + 1]) {
                unsigned char tmp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = tmp;
            }
        }
    }

    output[y * width + x] = window[4];
}

// BMP loading/saving
std::vector<unsigned char> loadGrayscaleBMP(const std::string& filename, int& width, int& height) {
    BMP image;
    if (!image.ReadFromFile(filename.c_str())) {
        throw std::runtime_error("Failed to read BMP file: " + filename);
    }

    width = image.TellWidth();
    height = image.TellHeight();

    std::vector<unsigned char> gray(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            RGBApixel pixel = image.GetPixel(x, y);

            // Convert to grayscale if needed
            unsigned char value = static_cast<unsigned char>(
                0.299 * pixel.Red + 0.587 * pixel.Green + 0.114 * pixel.Blue
                );

            gray[y * width + x] = value;
        }
    }

    return gray;
}

void saveGrayscaleBMP(const std::string& filename,
    const std::vector<unsigned char>& data,
    int width,
    int height) {
    BMP image;
    image.SetSize(width, height);
    image.SetBitDepth(24);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char value = data[y * width + x];

            RGBApixel pixel;
            pixel.Red = value;
            pixel.Green = value;
            pixel.Blue = value;
            pixel.Alpha = 0;

            image.SetPixel(x, y, pixel);
        }
    }

    if (!image.WriteToFile(filename.c_str())) {
        throw std::runtime_error("Failed to write BMP file: " + filename);
    }
}


int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <input.bmp> <output.bmp>\n";
            return 1;
        }

        const std::string inputFile = argv[1];
        const std::string outputFile = argv[2];

        int width = 0;
        int height = 0;

        std::vector<unsigned char> h_input = loadGrayscaleBMP(inputFile, width, height);
        std::vector<unsigned char> h_output(width * height);

        // Allocate CUDA array for texture
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
        cudaArray_t cuArray;
        CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

        // Copy input image to CUDA array
        CUDA_CHECK(cudaMemcpy2DToArray(
            cuArray,
            0, 0,
            h_input.data(),
            width * sizeof(unsigned char),
            width * sizeof(unsigned char),
            height,
            cudaMemcpyHostToDevice
        ));

        // Texture resource description
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Texture description
        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaTextureObject_t texObj = 0;
        CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

        // Output buffer
        unsigned char* d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(unsigned char)));

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
            (height + blockSize.y - 1) / blockSize.y);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        medianFilterKernel << <gridSize, blockSize >> > (texObj, d_output, width, height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        CUDA_CHECK(cudaMemcpy(h_output.data(),
            d_output,
            width * height * sizeof(unsigned char),
            cudaMemcpyDeviceToHost));

        saveGrayscaleBMP(outputFile, h_output, width, height);

        std::cout << "GPU processing time: " << ms << " ms\n";
        std::cout << "Output saved to: " << outputFile << "\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaDestroyTextureObject(texObj));
        CUDA_CHECK(cudaFreeArray(cuArray));
        CUDA_CHECK(cudaFree(d_output));

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}