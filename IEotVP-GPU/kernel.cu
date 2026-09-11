#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <locale.h>
#include <chrono>

using namespace std;

constexpr int THREADS_PER_BLOCK = 256;

size_t codeLength = 1900;  // n
size_t infoLength = 1280;  // k

constexpr int BITS = 64;

inline size_t wordsPerRow(size_t n) {
	return (n + BITS - 1) / BITS;
}

__host__ __device__ inline int getBit(const uint64_t* row, int bit) {
	return (int)((row[bit / BITS] >> (bit % BITS)) & 1ULL);
}

__host__ __device__ inline void setBit(uint64_t* row, int bit, int val) {
	const uint64_t mask = 1ULL << (bit%BITS);
	if (val) row[bit / BITS] |= mask;
	else row[bit / BITS] &= ~mask;
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(err) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while (0)


__global__ void findPivotKernel(
	const uint64_t* L,
	int col,
	int wordsCount,
	int wpr,
	int* pivotRow)
{
	int row = blockIdx.x *blockDim.x + threadIdx.x;
	if (row >= wordsCount) return;

	const uint64_t* rowPtr = L + (size_t)row * wpr;
	if (getBit(rowPtr, col)) {
		atomicMin(pivotRow, row);
	}
}

__device__ void xorTailFast(
	uint64_t* row,
	const uint64_t* base,
	int col,
	int n)   // n кратно 64, у тебя 1920
{
	int start = col + 1;
	if (start >= n) return;

	int startWord = start / 64;
	int startOff = start % 64;
	int nWords = n / 64;   // 30 для n=1920

	// первое частичное слово
	if (startOff != 0) {
		uint64_t mask = ~((1ULL << startOff) - 1ULL); // биты startOff..63
		row[startWord] ^= base[startWord] & mask;
		startWord++;
	}

	// полные слова до конца
	for (int w = startWord; w < nWords; w++) {
		row[w] ^= base[w];
	}
}

__global__ void updateGTmpKernel(
	uint64_t* G_tmp,
	const uint64_t* base,
	int col,
	int n,
	int wpr)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= n) return;

	uint64_t* rowPtr = G_tmp + (size_t)row * wpr;
	if (!getBit(rowPtr, col)) return;

	xorTailFast(rowPtr, base, col, n);
	// или xorTailPacked(rowPtr, base, col, n, wpr);
}

__global__ void eliminateColumnKernel(
	uint64_t* L,
	const uint64_t* base,
	int col,
	int wordsCount,
	int n,
	int wpr)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= wordsCount) return;

	uint64_t* rowPtr = L + (size_t)row * wpr;
	if (!getBit(rowPtr, col)) return;

	xorTailFast(rowPtr, base, col, n);
}

uint64_t* createIdentityPacked(size_t n) {
	size_t wpr = wordsPerRow(n);
	uint64_t* mat = new uint64_t[n * wpr];
	memset(mat, 0, n * wpr * sizeof(uint64_t));
	for (size_t i = 0; i < n; i++) {
		setBit(mat + i * wpr, (int)i, 1);
	}
	return mat;
}

bool ReadCodeWords(const string& filename, size_t codeLength, uint64_t*& L, size_t& wordsCount) {
	ifstream file(filename, ios::binary);
	if (!file.is_open()) {
		cout << "Не удалось открыть файл" << endl;
		return false;
	}

	file.seekg(0, ios::end);
	size_t fileSizeBytes = (size_t)file.tellg();
	file.seekg(0, ios::beg);

	size_t totalBits = fileSizeBytes * 8;
	wordsCount = totalBits / codeLength;
	if (wordsCount == 0) {
		cout << "Недостаточно данных в файле" << endl;
		file.close();
		return false;
	}

	cout << "Кодовых слов: " << wordsCount << ", n = " << codeLength << endl;

	uint8_t* buffer = new uint8_t[fileSizeBytes];
	file.read(reinterpret_cast<char*>(buffer), fileSizeBytes);
	file.close();

	const size_t wpr = wordsPerRow(codeLength);
	L = new uint64_t[wordsCount * wpr];
	memset(L, 0, wordsCount * wpr * sizeof(uint64_t));

	size_t bitPos = 0;
	for (size_t w = 0; w < wordsCount; w++) {
		uint64_t* row = L + w * wpr;
		for (size_t b = 0; b < codeLength; b++) {
			size_t byteIndex = bitPos / 8;
			size_t bitIndex = bitPos % 8;
			int bit = (buffer[byteIndex] >> bitIndex) & 1;
			if (bit) setBit(row, (int)b, 1);
			bitPos++;
		}
	}

	delete[] buffer;
	return true;
}

void WriteResultPackedToBin(const string& path, const uint64_t* G, size_t n, size_t wpr) {
	size_t totalBits = n * n;
	size_t totalBytes = (totalBits + 7) / 8;
	uint8_t* buf = new uint8_t[totalBytes]();

	size_t outBit = 0;
	for (size_t i = 0; i < n; i++) {
		const uint64_t* row = G + i * wpr;
		for (size_t j = 0; j < n; j++) {
			if (getBit(row, (int)j))
				buf[outBit / 8] |= (uint8_t)(1u << (outBit % 8));
			outBit++;
		}
	}
	ofstream out(path, ios::binary);
	out.write(reinterpret_cast<char*>(buf), totalBytes);
	delete[] buf;
}

// Основной алгоритм
void mainFunction(
	size_t wordsCount,
	size_t codeLength,
	size_t infoLength,
	uint64_t* d_L,
	uint64_t* d_G_tmp,
	int* d_pivot,
	uint64_t* d_base,
	int wpr)
{
	int blocks = (int)((wordsCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
	int gBlocks = (int)((codeLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

	for (int col = 0; col < (int)infoLength; col++) {
		int h_pivot = (int)wordsCount;
		CUDA_CHECK(cudaMemcpy(d_pivot, &h_pivot, sizeof(int), cudaMemcpyHostToDevice));

		findPivotKernel << <blocks, THREADS_PER_BLOCK >> > (
			d_L, col, (int)wordsCount, wpr, d_pivot);
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(&h_pivot, d_pivot, sizeof(int), cudaMemcpyDeviceToHost));
		if (h_pivot >= (int)wordsCount) continue;

		CUDA_CHECK(cudaMemcpy(
			d_base,
			d_L + (size_t)h_pivot * wpr,
			(size_t)wpr * sizeof(uint64_t),
			cudaMemcpyDeviceToDevice));

		updateGTmpKernel << <gBlocks, THREADS_PER_BLOCK >> > (
			d_G_tmp, d_base, col, (int)codeLength, wpr);

		eliminateColumnKernel << <blocks, THREADS_PER_BLOCK >> > (
			d_L, d_base, col, (int)wordsCount, (int)codeLength, wpr);

		CUDA_CHECK(cudaDeviceSynchronize());
	}
}

int main() {
	setlocale(LC_ALL, "");

	auto start = chrono::high_resolution_clock::now();

	string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\4.4.bin)";
	string OutputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\output.bin)";
	bool isBis = false;

	uint64_t* h_L = nullptr;
	uint64_t* d_L = nullptr;
	uint64_t* d_G_tmp = nullptr;
	uint64_t* d_base = nullptr;
	int* d_pivot = nullptr;

	size_t wordsCount = 0;
	if (!ReadCodeWords(InputFileName, codeLength, h_L, wordsCount)) {
		return 1;
	}

	size_t wpr = wordsPerRow(codeLength);
	size_t L_words = wordsCount * wpr;

	size_t L_bytes = L_words * sizeof(uint64_t);
	size_t G_bytes = codeLength * wpr * sizeof(uint64_t);
	uint64_t* G_tmp = createIdentityPacked(codeLength);

	CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
	CUDA_CHECK(cudaMalloc(&d_G_tmp, G_bytes));
	CUDA_CHECK(cudaMemcpy(d_L, h_L, L_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc(&d_base, wpr * sizeof(uint64_t)));
	CUDA_CHECK(cudaMalloc(&d_pivot, sizeof(int)));
	CUDA_CHECK(cudaMemcpy(d_G_tmp, G_tmp, G_bytes, cudaMemcpyHostToDevice));

	mainFunction(
		wordsCount,
		codeLength,
		infoLength,
		d_L,
		d_G_tmp,
		d_pivot,
		d_base,
		wpr
	);

	CUDA_CHECK(cudaMemcpy(G_tmp, d_G_tmp, G_bytes, cudaMemcpyDeviceToHost));
	WriteResultPackedToBin(OutputFileName, G_tmp, codeLength, wpr);

	// Освобождение
	CUDA_CHECK(cudaFree(d_L));
	CUDA_CHECK(cudaFree(d_G_tmp));
	CUDA_CHECK(cudaFree(d_pivot));
	CUDA_CHECK(cudaFree(d_base));

	delete[] h_L;
	delete[] G_tmp;

	auto ms = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count();
	cout << ms << endl;

	return 0;
}