#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <assert.h>

#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Compile with: nvcc -o guided_filter.dll guided_filter.cu -shared

/**
The guided filter works by solving a local linear model: q = aI + b where:

- I is the guidance image (in this case, the input image itself)
- a and b are linear coefficients
- q is the filtered output
**/

// Structure to hold filter parameters
struct GuidedFilterParams {
    float epsilon;      // Regularization parameter
    int radius;         // Filter window radius
    int width;          // Image width
    int height;         // Image height
    int numChannels;    // Number of color channels (e.g., 3 for RGB)
};

// Mean filter kernel (for both guidance image and input image)
__global__ void MeanFilterKernel(
    const unsigned char* input,
    float* output,
    const GuidedFilterParams params) 
{
    // -- Shared memory --
    // Populate shared memory for each block that contains all pixels in the block + border (based on radius)
    extern __shared__ unsigned char sharedMem[];

    const int sharedMemWidth = blockDim.x + (2 * params.radius);
    const int sharedMemHeight = blockDim.y + (2 * params.radius);    
    
    // Grid stride loop to cover the halo regions
    for(int dy = threadIdx.y; dy < sharedMemHeight; dy += blockDim.y) {
        for(int dx = threadIdx.x; dx < sharedMemWidth; dx += blockDim.x) {
            // Calculate global image coordinates
            int imgX = (blockIdx.x * blockDim.x) + dx - params.radius;
            int imgY = (blockIdx.y * blockDim.y) + dy - params.radius;

            // Clamp coordinates to image boundaries
            imgX = max(0, min(imgX, params.width - 1));
            imgY = max(0, min(imgY, params.height - 1));

            // Load RGB values (3 channels)
            const int sharedIdx = (dy * sharedMemWidth + dx) * params.numChannels;
            const int imgIdx = (imgY * params.width + imgX) * params.numChannels;

            sharedMem[sharedIdx] = input[imgIdx];          // R
            sharedMem[sharedIdx + 1] = input[imgIdx + 1];  // G
            sharedMem[sharedIdx + 2] = input[imgIdx + 2];  // B
        }
    }

    // Make sure all threads have loaded their data into shared memory
    __syncthreads();
    
    // -- Mean filter computation --
    int pixelIdx_X = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelIdx_Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixelIdx_X >= params.width || pixelIdx_Y >= params.height) return;

    // 1. Calculate thread indices    
    float sum_R = 0.0f;
    float sum_G = 0.0f;
    float sum_B = 0.0f;
    int windowSize = (2 * params.radius + 1) * (2 * params.radius + 1);

    // 2. For each pixel in window (radius):
    for (int dy = -params.radius; dy <= params.radius; ++dy) 
    {
        for (int dx = -params.radius; dx <= params.radius; ++dx) 
        {
            int sharedMem_x = threadIdx.x + dx + params.radius;
            int sharedMem_y = threadIdx.y + dy + params.radius;
            int shared_idx = (sharedMem_y * sharedMemWidth + sharedMem_x) * params.numChannels;

            // Sum over all channels
            sum_R += static_cast<float>(sharedMem[shared_idx]);
            sum_G += static_cast<float>(sharedMem[shared_idx + 1]);
            sum_B += static_cast<float>(sharedMem[shared_idx + 2]);
        }
    }

    // 3. Compute mean values
    const int outputIdx = (pixelIdx_Y * params.width + pixelIdx_X) * params.numChannels;
    output[outputIdx] = sum_R / windowSize;
    output[outputIdx + 1] = sum_G / windowSize;
    output[outputIdx + 2] = sum_B / windowSize;
}

// Compute variance kernel (since guide image = input image)
__global__ void ComputeCovarianceKernel(
    const float* mean_I,    // mean of image from previous kernel
    const unsigned char* I,  // input image
    float* var_I,          // output variance
    const GuidedFilterParams params) 
{
    // -- Shared memory --
    // Populate shared memory for each block that contains all pixels in the block + border (based on radius)
    extern __shared__ unsigned char sharedMem[];

    const int sharedMemWidth = blockDim.x + (2 * params.radius);
    const int sharedMemHeight = blockDim.y + (2 * params.radius);    
    
    // Grid stride loop to cover the halo regions
    for(int dy = threadIdx.y; dy < sharedMemHeight; dy += blockDim.y) {
        for(int dx = threadIdx.x; dx < sharedMemWidth; dx += blockDim.x) {
            // Calculate global image coordinates
            int imgX = (blockIdx.x * blockDim.x) + dx - params.radius;
            int imgY = (blockIdx.y * blockDim.y) + dy - params.radius;

            // Clamp coordinates to image boundaries
            imgX = max(0, min(imgX, params.width - 1));
            imgY = max(0, min(imgY, params.height - 1));

            // Load RGB values (3 channels)
            const int sharedIdx = (dy * sharedMemWidth + dx) * params.numChannels;  // Always 3 for RGB
            const int imgIdx = (imgY * params.width + imgX) * params.numChannels;

            sharedMem[sharedIdx] = I[imgIdx];          // R
            sharedMem[sharedIdx + 1] = I[imgIdx + 1];  // G
            sharedMem[sharedIdx + 2] = I[imgIdx + 2];  // B
        }
    }

    // Make sure all threads have loaded their data into shared memory
    __syncthreads();
    
    // -- variance computation --
    int pixelIdx_X = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelIdx_Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixelIdx_X >= params.width || pixelIdx_Y >= params.height) return;

    // Calculate mean of squared values
    float mean_I2_R = 0.0f;  // For storing mean of I*I
    float mean_I2_G = 0.0f;
    float mean_I2_B = 0.0f;
    int windowSize = (2 * params.radius + 1) * (2 * params.radius + 1);

    // For each pixel in window
    for (int dy = -params.radius; dy <= params.radius; ++dy) 
    {
        for (int dx = -params.radius; dx <= params.radius; ++dx) 
        {
            int sharedMem_x = threadIdx.x + dx + params.radius;
            int sharedMem_y = threadIdx.y + dy + params.radius;
            int shared_idx = (sharedMem_y * sharedMemWidth + sharedMem_x) * params.numChannels;

            // Calculate mean(I*I)
            mean_I2_R += static_cast<float>(sharedMem[shared_idx]) * static_cast<float>(sharedMem[shared_idx]);
            mean_I2_G += static_cast<float>(sharedMem[shared_idx + 1]) * static_cast<float>(sharedMem[shared_idx + 1]);
            mean_I2_B += static_cast<float>(sharedMem[shared_idx + 2]) * static_cast<float>(sharedMem[shared_idx + 2]);
        }
    }

    // Compute mean of squared values
    mean_I2_R /= windowSize;
    mean_I2_G /= windowSize;
    mean_I2_B /= windowSize;

    // Compute variance = mean(I*I) - mean(I)*mean(I)
    const int outputIdx = (pixelIdx_Y * params.width + pixelIdx_X) * params.numChannels;
    var_I[outputIdx] = mean_I2_R - (mean_I[outputIdx] * mean_I[outputIdx]);
    var_I[outputIdx + 1] = mean_I2_G - (mean_I[outputIdx + 1] * mean_I[outputIdx + 1]);
    var_I[outputIdx + 2] = mean_I2_B - (mean_I[outputIdx + 2] * mean_I[outputIdx + 2]);
}

// Compute coefficient kernel
__global__ void ComputeCoefficientsKernel(
    const float* var_I,      // variance from previous kernel
    const float* mean_I,     // mean values from first kernel
    float* a,                // output a coefficient
    float* b,                // output b coefficient
    const GuidedFilterParams params
) {
    // Calculate pixel position
    const int pixelIdx_X = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixelIdx_Y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if this thread should process a pixel
    if (pixelIdx_X >= params.width || pixelIdx_Y >= params.height) return;

    // Calculate index for this pixel's RGB values
    const int idx = (pixelIdx_Y * params.width + pixelIdx_X) * params.numChannels;

    // For each color channel
    for (int ch = 0; ch < params.numChannels; ch++) {
        // Calculate a = var_I / (var_I + epsilon)
        // Higher variance (edges) → a ≈ 1 (preserve detail)
        // Lower variance (flat areas) → a ≈ 0 (more smoothing)
        a[idx + ch] = var_I[idx + ch] / (var_I[idx + ch] + params.epsilon);
        
        // Calculate b = mean_I - a * mean_I
        // This ensures the filter preserves the mean value in each window
        b[idx + ch] = mean_I[idx + ch] * (1.0f - a[idx + ch]);
    }
}

// Final guided filter kernel
__global__ void GuidedFilterKernel(
    const float* a,           // a coefficients from previous kernel
    const float* b,           // b coefficients from previous kernel
    const unsigned char* I,   // original input image
    unsigned char* output,    // final filtered output
    const GuidedFilterParams params
) {
    // -- Shared memory setup --
    extern __shared__ unsigned char sharedMem[];

    const int sharedMemWidth = blockDim.x + (2 * params.radius);
    const int sharedMemHeight = blockDim.y + (2 * params.radius);    
    
    // Load input image into shared memory including halo region
    for(int dy = threadIdx.y; dy < sharedMemHeight; dy += blockDim.y) {
        for(int dx = threadIdx.x; dx < sharedMemWidth; dx += blockDim.x) {
            // Calculate global image coordinates
            int imgX = (blockIdx.x * blockDim.x) + dx - params.radius;
            int imgY = (blockIdx.y * blockDim.y) + dy - params.radius;

            // Clamp coordinates to image boundaries
            imgX = max(0, min(imgX, params.width - 1));
            imgY = max(0, min(imgY, params.height - 1));

            // Load RGB values
            const int sharedIdx = (dy * sharedMemWidth + dx) * params.numChannels;
            const int imgIdx = (imgY * params.width + imgX) * params.numChannels;

            sharedMem[sharedIdx] = I[imgIdx];        // R
            sharedMem[sharedIdx + 1] = I[imgIdx + 1];  // G
            sharedMem[sharedIdx + 2] = I[imgIdx + 2];  // B
        }
    }

    __syncthreads();

    // Calculate pixel position
    const int pixelIdx_X = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixelIdx_Y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pixelIdx_X >= params.width || pixelIdx_Y >= params.height) return;

    // Initialize accumulators for each channel
    float sum_R = 0.0f;
    float sum_G = 0.0f;
    float sum_B = 0.0f;
    const int windowSize = (2 * params.radius + 1) * (2 * params.radius + 1);

    // Calculate mean of q = aI + b in the window
    for (int dy = -params.radius; dy <= params.radius; ++dy) {
        for (int dx = -params.radius; dx <= params.radius; ++dx) {
            // Calculate position in shared memory
            int sharedMem_x = threadIdx.x + dx + params.radius;
            int sharedMem_y = threadIdx.y + dy + params.radius;
            int shared_idx = (sharedMem_y * sharedMemWidth + sharedMem_x) * params.numChannels;

            // Get coefficients for current window position
            int window_x = pixelIdx_X + dx;
            int window_y = pixelIdx_Y + dy;
            window_x = max(0, min(window_x, params.width - 1));
            window_y = max(0, min(window_y, params.height - 1));
            int coef_idx = (window_y * params.width + window_x) * params.numChannels;

            // Apply q = aI + b for each channel
            sum_R += a[coef_idx] * static_cast<float>(sharedMem[shared_idx]) + b[coef_idx];
            sum_G += a[coef_idx + 1] * static_cast<float>(sharedMem[shared_idx + 1]) + b[coef_idx + 1];
            sum_B += a[coef_idx + 2] * static_cast<float>(sharedMem[shared_idx + 2]) + b[coef_idx + 2];
        }
    }

    // Average the results
    sum_R /= windowSize;
    sum_G /= windowSize;
    sum_B /= windowSize;

    // Write final result to output, ensuring values are in [0, 255]
    const int outputIdx = (pixelIdx_Y * params.width + pixelIdx_X) * params.numChannels;
    output[outputIdx] = static_cast<unsigned char>(max(0.0f, min(255.0f, sum_R)));
    output[outputIdx + 1] = static_cast<unsigned char>(max(0.0f, min(255.0f, sum_G)));
    output[outputIdx + 2] = static_cast<unsigned char>(max(0.0f, min(255.0f, sum_B)));
}

// Main function to be called externally
extern "C" __declspec(dllexport) void ApplyGuidedFilter(
    const unsigned char* h_input,
    unsigned char* h_output,
    int width,
    int height,
    float epsilon,
    int radius, 
    int numChannels
) {
    // Setup filter parameters
    GuidedFilterParams params;
    params.width = width;
    params.height = height;
    params.epsilon = epsilon;
    params.radius = radius;
    params.numChannels = numChannels; // Assuming RGB images

    // 1. Allocate device memory
    unsigned char *d_input, *d_output;
    int charImageSizeInBytes = width * height * numChannels * sizeof(unsigned char);
    checkCudaErrors(cudaMallocManaged(&d_input, charImageSizeInBytes));
    checkCudaErrors(cudaMallocManaged(&d_output, charImageSizeInBytes));

    float *d_mean, *d_var, *d_a, *d_b;
    int floatImageSizeInBytes = width * height * numChannels * sizeof(float);
    checkCudaErrors(cudaMallocManaged(&d_mean, floatImageSizeInBytes)); // mean image
    checkCudaErrors(cudaMallocManaged(&d_var, floatImageSizeInBytes));  // variance image
    checkCudaErrors(cudaMallocManaged(&d_a, floatImageSizeInBytes));    // a coefficients
    checkCudaErrors(cudaMallocManaged(&d_b, floatImageSizeInBytes));    // b coefficients

    // 2. Copy input data to device
    checkCudaErrors(cudaMemcpy(d_input, h_input, charImageSizeInBytes, cudaMemcpyHostToDevice));

    // 3. Set up grid and block dimensions
    int numThreads = 16;
    dim3 blockDim(numThreads, numThreads);
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x, 
        (height + blockDim.y - 1) / blockDim.y
    );

    // 3.5. Set up shared memory size
    int x_dim = blockDim.x + (2 * radius);
    int y_dim = blockDim.y + (2 * radius);
    int sharedMemSizeInBytes = x_dim * y_dim * numChannels * sizeof(unsigned char);

    // 4. Launch kernels in sequence:
    // - meanFilterKernel
    MeanFilterKernel<<<gridDim, blockDim, sharedMemSizeInBytes>>>(d_input, d_mean, params);
    checkCudaErrors(cudaDeviceSynchronize());

    // - computeCovarianceKernel
    ComputeCovarianceKernel<<<gridDim, blockDim, sharedMemSizeInBytes>>>(d_mean, d_input, d_var, params);
    checkCudaErrors(cudaDeviceSynchronize());

    // - computeCoefficientsKernel
    ComputeCoefficientsKernel<<<gridDim, blockDim>>>(d_var, d_mean, d_a, d_b, params);
    checkCudaErrors(cudaDeviceSynchronize());

    // - guidedFilterKernel
    GuidedFilterKernel<<<gridDim, blockDim, sharedMemSizeInBytes>>>(d_a, d_b, d_input, d_output, params);
    checkCudaErrors(cudaDeviceSynchronize());

    // 5. Copy result back to host
    checkCudaErrors(cudaMemcpy(h_output, d_output, charImageSizeInBytes, cudaMemcpyDeviceToHost));

    // 6. Clean up device memory
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_mean));
    checkCudaErrors(cudaFree(d_var));
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
}
