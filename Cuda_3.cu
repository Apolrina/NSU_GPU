#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <unistd.h>
#include <thread>

#define CUDA_DEBUG

#ifdef CUDA_DEBUG
#define CUDA_CHECK_ERROR(err)           \
if (err != cudaSuccess) {          \
    printf("Cuda error: %s\n", cudaGetErrorString(err));    \
    printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);  \
}
#else
#define CUDA_CHECK_ERROR(err)
#endif

#define GAUSS(x, sigma) \
1 / sqrt(2 * M_PI * sigma * sigma) * exp(-x * x / (2 * sigma * sigma))

#define BLOCK_SIZE 16
#define FILTER_SIZE 3
#define TILE_SIZE (BLOCK_SIZE - FILTER_SIZE + 1)
#define M_PI 3.14159265358979323846


__global__ void blurFilterShared(const unsigned char* input, unsigned char* output, int width, int height, int channels) {

    extern __shared__ unsigned char sharedMem[];
    int sharedWidth = blockDim.x + FILTER_SIZE - 1;
    int sharedHeight = blockDim.y + FILTER_SIZE - 1;

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    int sharedX = threadIdx.x + FILTER_SIZE / 2;
    int sharedY = threadIdx.y + FILTER_SIZE / 2;

    float sigma = 1;
    float kernel_weight = 0;


    for (int dy = -FILTER_SIZE / 2; dy <= FILTER_SIZE / 2; dy++) {
      for (int dx = -FILTER_SIZE / 2; dx <= FILTER_SIZE / 2; dx++) {
        int sharedMemX = sharedX + dx;
        int sharedMemY = sharedY + dy;
        int globalX = min(max(x + dx, 0), width - 1);
        int globalY = min(max(y + dy, 0), height - 1);
        for(int c = 0; c < channels; c++) {
          sharedMem[(sharedMemY * sharedWidth + sharedMemX) * channels + c] = input[(globalY * width + globalX) * channels + c];
        }
        kernel_weight += GAUSS(dx, sigma) * GAUSS(dy, sigma);
      }
    }

    __syncthreads();

    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            int color = 0;
            for (int dy = -FILTER_SIZE / 2; dy <= FILTER_SIZE / 2; dy++) {
                for (int dx = -FILTER_SIZE / 2; dx <= FILTER_SIZE / 2; dx++) {
                    int shiftedX = sharedX + dx;
                    int shiftedY = sharedY + dy;

                    float factor = GAUSS(dx, sigma) * GAUSS(dy, sigma) / kernel_weight;

                    if (shiftedX >= 0 && shiftedX < sharedWidth && shiftedY >= 0 && shiftedY < sharedHeight) {
                        color += factor * sharedMem[(shiftedY * sharedWidth + shiftedX) * channels + c];
                    }
                }
            }

            // Запись результата обратно в глобальную память
            output[(y * width + x) * channels + c] = color;
        }

    }
}

template <typename KernelFunc, typename... Args>
void measureKernelExecutionTime(const char* message, KernelFunc kernel, dim3 gridSize, dim3 blockSize, size_t sharedMemSize, Args... args) {
    // Запуск таймера
    auto start = std::chrono::high_resolution_clock::now();
    // Запуск ядра
    kernel<<<gridSize, blockSize, sharedMemSize>>>(args...);
    // Синхронизация устройства
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    // Остановка таймера
    auto end = std::chrono::high_resolution_clock::now();
    // Вывод времени выполнения в микросекундах
    std::cout << message << ": " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
              << " mcs" << std::endl;
}

void loadImage(const std::string& filename, cv::Mat& image) {
    image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        std::cerr << "Image is empty " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Image is loaded. Size: " << image.cols << "x" << image.rows << ", Channels num: " << image.channels() << std::endl;
}

void saveImage(const std::string& filename, const cv::Mat& image) {
    if (!cv::imwrite(filename, image)) {
        std::cerr << "The error during saving image " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Output image was saved successfully " << filename << std::endl;
}

// Выделение памяти и копирование данных на GPU
void allocateAndCopyToDevice(unsigned char* h_data, unsigned char** d_data, size_t size) {
    CUDA_CHECK_ERROR(cudaMalloc(d_data, size));
    CUDA_CHECK_ERROR(cudaMemcpy(*d_data, h_data, size, cudaMemcpyHostToDevice));
}

// Основная функция обработки на GPU
void processOnGPU(int deviceID, int numGPUs, unsigned char* h_inputImage, unsigned char* h_outputImage, 
                  int width, int height, int channels) {
    cudaSetDevice(deviceID);

    int overlap = FILTER_SIZE / 2;
    int segmentHeight = height / numGPUs;
    int yOffset = segmentHeight * deviceID;
    
    if (deviceID == numGPUs - 1) {
        segmentHeight += height % numGPUs;
    }

    int startRow = (deviceID == 0) ? 0 : yOffset - overlap;
    int endRow = (deviceID == numGPUs - 1) ? height : yOffset + segmentHeight + overlap;
   
    int segmentHeightWithOverlap = endRow - startRow;

    size_t segmentSizeWithOverlap = width * segmentHeightWithOverlap * channels * sizeof(unsigned char);
    unsigned char *d_inputImage, *d_outputImage;

    cudaMalloc(&d_inputImage, segmentSizeWithOverlap);
    cudaMalloc(&d_outputImage, segmentSizeWithOverlap);

    cudaMemcpy(d_inputImage, &h_inputImage[startRow * width * channels], segmentSizeWithOverlap, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (segmentHeightWithOverlap + TILE_SIZE - 1) / TILE_SIZE);
    size_t sharedMemSize = (BLOCK_SIZE + FILTER_SIZE - 1) * (BLOCK_SIZE + FILTER_SIZE - 1) * channels * sizeof(unsigned char);

    std::string gpuMessage = "Shared: GPU " + std::to_string(deviceID) + " execution time";
    measureKernelExecutionTime(gpuMessage.c_str(), blurFilterShared, gridSize, blockSize, sharedMemSize, 
                               d_inputImage, d_outputImage, width, segmentHeightWithOverlap, channels);

    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    int copyStartRow = (deviceID == 0) ? 0 : overlap;
    int copyEndRow = (deviceID == numGPUs - 1) ? segmentHeightWithOverlap : segmentHeightWithOverlap - overlap;

    int copyHeight = copyEndRow - copyStartRow;

    size_t segmentSize = width * copyHeight * channels * sizeof(unsigned char);

    cudaMemcpy(&h_outputImage[yOffset * width * channels], &d_outputImage[copyStartRow * width * channels], segmentSize, cudaMemcpyDeviceToHost);
   
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}


// Основная функция обработки
void processSharedOnMultipleGPUs(const cv::Mat& image, cv::Mat& outputImage) {
    int numGPUs;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&numGPUs));
    if (numGPUs == 0) {
        std::cerr << "No CUDA-compatible devices found" << std::endl;
        return;
    }

    std::cout << "Number of GPUs available: " << numGPUs << std::endl;

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();

    unsigned char* h_inputImage = image.data; 
    unsigned char* h_outputImage = outputImage.data;
    
    std::vector<std::thread> threads;

    // Запуск обработки на каждом GPU в отдельных потоках
    for (int i = 0; i < numGPUs; i++) {
        threads.emplace_back(processOnGPU, i, numGPUs, h_inputImage, h_outputImage, width, height, channels);
    }

    // Ожидание завершения всех потоков
    for (auto& t : threads) {
        t.join();
    }
}

int main() {
    const std::string filename = "image.jpeg";
    cv::Mat image;
    loadImage(filename, image);

    cv::Mat sharedOutputImage = cv::Mat::zeros(image.size(), image.type());

    processSharedOnMultipleGPUs(image, sharedOutputImage);
    saveImage("./output_image_shared.png", sharedOutputImage);

    return 0;
}



