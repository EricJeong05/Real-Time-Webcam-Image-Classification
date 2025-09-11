#include <cuda_runtime.h>
#include <stdio.h>

// Compile with: nvcc -o edge_detect.dll edge_detect.cu -shared

// This kernel performs a simple edge detection on an input image. For each pixel,
// it calculates the intensity gradient in the x and y directions and combines them
// to produce an edge-detected output image.
__global__ void EdgeDetectKernel(
    unsigned char* inputImage, unsigned char* outputImage,
    int imageWidth, int imageHeight, int numChannels)
{
    // Calculate the global x and y coordinates of the current thread
    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check to ensure the thread is within the image dimensions
    if (pixelX >= imageWidth || pixelY >= imageHeight) return;

    // Calculate the linear index for the current pixel
    int pixelIndex = (pixelY * imageWidth + pixelX) * numChannels;

    // Process each color channel (e.g., R, G, B)
    for (int channel = 0; channel < numChannels; channel++) {
        int gradientX = 0, gradientY = 0;

        // Calculate the horizontal gradient (difference with the pixel to the right)
        if (pixelX + 1 < imageWidth) {
            gradientX = abs(int(inputImage[pixelIndex + channel]) - int(inputImage[pixelIndex + numChannels + channel]));
        }

        // Calculate the vertical gradient (difference with the pixel below)
        if (pixelY + 1 < imageHeight) {
            gradientY = abs(int(inputImage[pixelIndex + channel]) - int(inputImage[pixelIndex + imageWidth * numChannels + channel]));
        }

        // Combine the gradients and clamp the value to the 0-255 range
        outputImage[pixelIndex + channel] = (unsigned char)min(255, gradientX + gradientY);
    }
}

// =================================================================================
// Host Function: run_edge_detect
//
// This function is the entry point for running the edge detection process. It
// handles memory allocation on the GPU, data transfers between the host and
// device, kernel launch, and final cleanup. This function is exported as a DLL
// to be callable from other applications.
// =================================================================================
extern "C" __declspec(dllexport) void RunEdgeDetect(
    unsigned char* h_input, unsigned char* h_output,
    int imageWidth, int imageHeight, int numChannels)
{
    unsigned char *d_input, *d_output;
    size_t imageSize = imageWidth * imageHeight * numChannels * sizeof(unsigned char);

    // Allocate Unified Memory for the input and output images on the GPU.
    // Unified Memory is accessible from both the CPU and GPU.
    cudaMallocManaged(&d_input, imageSize);
    cudaMallocManaged(&d_output, imageSize);

    // Copy the input image data from the host (CPU) to the managed memory (GPU)
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

    // Define the dimensions of the thread blocks and the grid
    dim3 blockDim(16, 16);
    dim3 gridDim((imageWidth + blockDim.x - 1) / blockDim.x,
                 (imageHeight + blockDim.y - 1) / blockDim.y);

    // Launch the edge detection kernel on the GPU
    EdgeDetectKernel<<<gridDim, blockDim>>>(d_input, d_output, imageWidth, imageHeight, numChannels);

    // Synchronize the CPU and GPU to ensure the kernel has finished execution
    cudaDeviceSynchronize();

    // Copy the resulting edge-detected image from the GPU back to the host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Free the allocated memory on the GPU
    cudaFree(d_input);
    cudaFree(d_output);
}