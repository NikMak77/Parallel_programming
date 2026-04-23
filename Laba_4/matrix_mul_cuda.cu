#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>

const int WARMUP_RUNS = 3;
const int TIMED_RUNS = 10;

std::vector<double> read_matrix_flat(const std::string& filename, int& n) {
    std::ifstream file(filename);
    std::vector<double> data;
    std::string line;
    int rows = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        double val;
        int cols = 0;
        while (ss >> val) {
            data.push_back(val);
            ++cols;
        }
        if (cols > 0) {
            ++rows;
            if (n == 0) n = cols;
            else if (n != cols) {
                std::cerr << "Ошибка: матрица не квадратная\n";
                exit(1);
            }
        }
    }
    if (rows != n) {
        std::cerr << "Ошибка: число строк не равно числу столбцов\n";
        exit(1);
    }
    return data;
}

void write_matrix_flat(const std::string& filename, const std::vector<double>& data, int n) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(6);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << data[i * n + j];
            if (j < n - 1) file << " ";
        }
        file << "\n";
    }
}

template <int BLOCK_SIZE>
__global__ void matrixMulKernel(const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = n * BLOCK_SIZE * by;
    int aEnd = aBegin + n - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * n;

    double Csub = 0.0;

    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep) {

        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + n * ty + tx];
        Bs[ty][tx] = B[b + n * ty + tx];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = n * (BLOCK_SIZE * by + ty) + (BLOCK_SIZE * bx + tx);
    if (c < n * n)
        C[c] = Csub;
}

void launchKernel(int block_size, dim3 grid, dim3 block,
    const double* d_A, const double* d_B, double* d_C, int n) {
    switch (block_size) {
    case 8:
        matrixMulKernel<8> << <grid, block >> > (d_A, d_B, d_C, n);
        break;
    case 16:
        matrixMulKernel<16> << <grid, block >> > (d_A, d_B, d_C, n);
        break;
    case 24:
        matrixMulKernel<24> << <grid, block >> > (d_A, d_B, d_C, n);
        break;
    case 32:
        matrixMulKernel<32> << <grid, block >> > (d_A, d_B, d_C, n);
        break;
    default:
        std::cerr << "Unsupported block size: " << block_size << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Использование: " << argv[0]
            << " <matrixA> <matrixB> <result> <block_size>\n";
        return 1;
    }

    std::string fileA = argv[1];
    std::string fileB = argv[2];
    std::string fileC = argv[3];
    int block_size = std::stoi(argv[4]);

    int n = 0;
    std::vector<double> h_A = read_matrix_flat(fileA, n);
    std::vector<double> h_B = read_matrix_flat(fileB, n);
    std::vector<double> h_C(n * n, 0.0);

    size_t bytes = n * n * sizeof(double);
    std::cout << "[DEBUG] n = " << n << ", bytes = " << bytes << std::endl;

    double* d_A, * d_B, * d_C;
    cudaError_t err;

    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Ошибка cudaMalloc d_A: код " << (int)err << std::endl;
        return 1;
    }
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Ошибка cudaMalloc d_B: код " << (int)err << std::endl;
        cudaFree(d_A);
        return 1;
    }
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) {
        std::cerr << "Ошибка cudaMalloc d_C: код " << (int)err << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }

    err = cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Ошибка копирования A: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return 1;
    }
    err = cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Ошибка копирования B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        return 1;
    }

    dim3 block(block_size, block_size);
    dim3 grid((n + block.x - 1) / block.x,
        (n + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        launchKernel(block_size, grid, block, d_A, d_B, d_C, n);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Ошибка после прогрева: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        return 1;
    }

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_RUNS; ++i) {
        launchKernel(block_size, grid, block, d_A, d_B, d_C, n);
    }
    cudaEventRecord(stop);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Ошибка после основного запуска: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        return 1;
    }

    float total_milliseconds = 0;
    cudaEventElapsedTime(&total_milliseconds, start, stop);
    double avg_seconds = (total_milliseconds / 1000.0) / TIMED_RUNS;

    err = cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Ошибка копирования результата: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        return 1;
    }

    write_matrix_flat(fileC, h_C, n);

    long long volume = (long long)n * n * n;
    std::cout << "Время выполнения (среднее за " << TIMED_RUNS << " запусков): "
        << std::scientific << std::setprecision(6) << avg_seconds << " секунд\n";
    std::cout << "Размер матрицы: " << n << " x " << n << "\n";
    std::cout << "Размер блока: " << block_size << " x " << block_size << "\n";
    std::cout << "Объем задачи: " << volume << " операций\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}