#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <locale.h>
#include <chrono>

using namespace std;

constexpr int THREADS_PER_BLOCK = 256;
constexpr int BITS = 8;  // 8 бит в одном uint8_t

inline size_t bytesPerRow(size_t n) {
	return (n + BITS - 1) / BITS;  // (n + 7) / 8
}

__host__ __device__ inline int getBit(const uint8_t* row, int bit) {
	return (row[bit / BITS] >> (bit % BITS)) & 1;
}

__host__ __device__ inline void setBit(uint8_t* row, int bit, int val) {
	const uint8_t mask = (uint8_t)(1u << (bit % BITS));
	if (val) row[bit / BITS] |= mask;
	else     row[bit / BITS] &= (uint8_t)~mask;
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

// Kernel-ы

__global__ void findPivotKernel(
	const uint8_t* L,
	int col,
	int wordsCount,
	int bpr,
	int* pivotRow)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= wordsCount) return;

	const uint8_t* rowPtr = L + (size_t)row * bpr;
	if (getBit(rowPtr, col))
		atomicMin(pivotRow, row);
}

__device__ void xorTailFast(
	uint8_t* row,
	const uint8_t* base,
	int col,
	int n)
{
	int start = col + 1;
	if (start >= n) return;

	int startByte = start / BITS;
	int startOff = start % BITS;
	int fullEnd = n / BITS;   // полные байты [0 .. fullEnd)
	int rem = n % BITS;   // хвост последнего байта

	// первое (частичное) байт
	if (startOff != 0) {
		uint8_t mask = (uint8_t)(0xFFu << startOff); // биты startOff..7

		int lastBitInByte = startByte * BITS + (BITS - 1);
		if (lastBitInByte >= n) {
			int keep = n - startByte * BITS;
			uint8_t hi = (keep == BITS) ? 0xFFu : (uint8_t)((1u << keep) - 1u);
			mask &= hi;
		}

		row[startByte] ^= (uint8_t)(base[startByte] & mask);
		startByte++;
	}

	// полные байты
	for (int b = startByte; b < fullEnd; b++)
		row[b] ^= base[b];

	// последний неполный байт
	if (rem != 0 && fullEnd >= startByte) {
		uint8_t mask = (uint8_t)((1u << rem) - 1u);
		row[fullEnd] ^= (uint8_t)(base[fullEnd] & mask);
	}
}

__global__ void updateGTmpKernel(
	uint8_t* G_tmp,
	const uint8_t* base,
	int col,
	int n,
	int bpr)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= n) return;

	uint8_t* rowPtr = G_tmp + (size_t)row * bpr;
	if (!getBit(rowPtr, col)) return;

	xorTailFast(rowPtr, base, col, n);
}

__global__ void eliminateColumnKernel(
	uint8_t* L,
	const uint8_t* base,
	int col,
	int wordsCount,
	int n,
	int bpr)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= wordsCount) return;

	uint8_t* rowPtr = L + (size_t)row * bpr;
	if (!getBit(rowPtr, col)) return;

	xorTailFast(rowPtr, base, col, n);
}

// Вспомогательные функции хоста

uint8_t* createIdentityPacked(size_t n) {
	size_t bpr = bytesPerRow(n);
	uint8_t* mat = new uint8_t[n * bpr];
	memset(mat, 0, n * bpr);
	for (size_t i = 0; i < n; i++)
		setBit(mat + i * bpr, (int)i, 1);
	return mat;
}

bool ReadCodeWords(
	const string& filename,
	size_t codeLength,
	uint8_t*& L,
	size_t& wordsCount,
	bool isBis,
	double mCoeff)
{
	ifstream file(filename, ios::binary);
	if (!file.is_open()) {
		cout << "Не удалось открыть файл" << endl;
		return false;
	}

	file.seekg(0, ios::end);
	size_t fileSizeBytes = (size_t)file.tellg();
	file.seekg(0, ios::beg);

	size_t fileBits = isBis ? fileSizeBytes : fileSizeBytes * 8;

	size_t wantBits = (size_t)(mCoeff * (double)codeLength * (double)codeLength);
	wantBits = (wantBits / codeLength) * codeLength;

	size_t totalBits = (wantBits < fileBits) ? wantBits : fileBits;
	totalBits = (totalBits / codeLength) * codeLength;
	wordsCount = totalBits / codeLength;

	if (wordsCount == 0) {
		cout << "Недостаточно данных в файле" << endl;
		file.close();
		return false;
	}

	size_t bytesToRead = isBis ? totalBits : (totalBits + 7) / 8;
	if (bytesToRead > fileSizeBytes)
		bytesToRead = fileSizeBytes;

	uint8_t* buffer = new uint8_t[bytesToRead];
	file.read(reinterpret_cast<char*>(buffer), (std::streamsize)bytesToRead);
	file.close();

	const size_t bpr = bytesPerRow(codeLength);
	L = new uint8_t[wordsCount * bpr];
	memset(L, 0, wordsCount * bpr);

	if (isBis) {
		// 1 байт файла = 1 бит
		size_t pos = 0;
		for (size_t w = 0; w < wordsCount; w++) {
			uint8_t* row = L + w * bpr;
			for (size_t b = 0; b < codeLength; b++) {
				if (pos < bytesToRead && buffer[pos] != 0)
					setBit(row, (int)b, 1);
				pos++;
			}
		}
	}
	else {
		// BIN
		size_t bitPos = 0;
		for (size_t w = 0; w < wordsCount; w++) {
			uint8_t* row = L + w * bpr;
			for (size_t b = 0; b < codeLength; b++) {
				size_t byteIndex = bitPos / 8;
				size_t bitIndex = bitPos % 8; // 0 = младший бит байта
				if (byteIndex < bytesToRead) {
					// эквивалент: buffer[byteIndex] & (1 << bitIndex)
					if ((buffer[byteIndex] >> bitIndex) & 1)
						setBit(row, (int)b, 1);
				}
				bitPos++;
			}
		}
		// memcpy при n%8==0 даёт то же только если тот же LSB-порядок;
		// для отладки лучше всегда этот путь (потом можно вернуть memcpy)
	}

	delete[] buffer;
	return true;
}

void WriteResultPacked(
	const string& path,
	const uint8_t* G,
	size_t n,
	size_t bpr,
	bool isBis)
{
	ofstream out(path, ios::binary);
	if (!out.is_open()) {
		cerr << "Не удалось открыть файл для записи" << endl;
		return;
	}

	if (isBis) {
		size_t totalBytes = n * n;
		uint8_t* buf = new uint8_t[totalBytes];
		size_t pos = 0;
		for (size_t i = 0; i < n; i++) {
			const uint8_t* row = G + i * bpr;
			for (size_t j = 0; j < n; j++)
				buf[pos++] = getBit(row, (int)j) ? 0xFF : 0x00;
		}
		out.write(reinterpret_cast<char*>(buf), totalBytes);
		delete[] buf;
	}
	else {
		size_t totalBits = n * n;
		size_t totalBytes = (totalBits + 7) / 8;
		uint8_t* buf = new uint8_t[totalBytes]();
		size_t outBit = 0;
		for (size_t i = 0; i < n; i++) {
			const uint8_t* row = G + i * bpr;
			for (size_t j = 0; j < n; j++) {
				if (getBit(row, (int)j))
					buf[outBit / 8] |= (uint8_t)(1u << (outBit % 8));
				outBit++;
			}
		}
		out.write(reinterpret_cast<char*>(buf), totalBytes);
		delete[] buf;
	}
}
// Основной циклитерационного исключения
void runIterativeEliminationGPU(
	size_t wordsCount, // число кодовых слов
	size_t codeLength, // длина слова
	size_t infoLength, // число шагов
	uint8_t* d_L, // матрица кодовых слов на ГПУ
	uint8_t* d_G_tmp, // накопленная Г на ГПУ
	int* d_pivot, // номер пивота строки на ГПУ
	uint8_t* d_base, // копия опорной строки на ГПУ
	int bpr) // bytes per row = ceil(n/8)
{
	int blocks = (int)((wordsCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK); // сколько блоков необходимо, что бы покрыть все строки Л
	int gBlocks = (int)((codeLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

	//создание streams
	cudaStream_t streamPivot, streamUpdate, streamElim;
	CUDA_CHECK(cudaStreamCreate(&streamPivot)); // сброс пивота, поискб копия 
	CUDA_CHECK(cudaStreamCreate(&streamUpdate)); // ядро updateGTmpKernel
	CUDA_CHECK(cudaStreamCreate(&streamElim)); // ядро eleminateColumnKernel

	int* h_pivot_pinned = nullptr;
	// pinned память pivot
	CUDA_CHECK(cudaMallocHost(&h_pivot_pinned, sizeof(int)));

	// Цикл по столбцам
	for (int col = 0; col < (int)infoLength; col++) {
		*h_pivot_pinned = (int)wordsCount; // если единиц в столбце нет, значение так и останется wordsCount, значит пропускаем столбец

		CUDA_CHECK(cudaMemcpyAsync(
			d_pivot, h_pivot_pinned, sizeof(int),
			cudaMemcpyHostToDevice, streamPivot));
		// Поиск пивотов
		findPivotKernel << <blocks, THREADS_PER_BLOCK, 0, streamPivot >> > ( // каждый поток смотрит свою строку
			d_L, col, (int)wordsCount, bpr, d_pivot);//как итог d_pivot - минномер строки с единицей

		// Переносим pivot на ЦПУ
		CUDA_CHECK(cudaMemcpyAsync(
			h_pivot_pinned, d_pivot, sizeof(int),
			cudaMemcpyDeviceToHost, streamPivot));

		CUDA_CHECK(cudaStreamSynchronize(streamPivot)); // пока Цпу не узнает pivot нельзя понять откуда копировать base

		int h_pivot = *h_pivot_pinned;
		if (h_pivot >= (int)wordsCount) // если нет опорной строки
			continue; // пропускаем шаг

		// Копируем base
		CUDA_CHECK(cudaMemcpyAsync(
			d_base,
			d_L + (size_t)h_pivot * bpr, // начало пивот строки в плоском Л
			(size_t)bpr * sizeof(uint8_t),
			cudaMemcpyDeviceToDevice,
			streamPivot));
		CUDA_CHECK(cudaStreamSynchronize(streamPivot)); // base готов

		// Два независимых ядра — параллельно
		updateGTmpKernel << <gBlocks, THREADS_PER_BLOCK, 0, streamUpdate >> > (
			d_G_tmp, d_base, col, (int)codeLength, bpr);

		eliminateColumnKernel << <blocks, THREADS_PER_BLOCK, 0, streamElim >> > (
			d_L, d_base, col, (int)wordsCount, (int)codeLength, bpr);

		CUDA_CHECK(cudaStreamSynchronize(streamUpdate));
		CUDA_CHECK(cudaStreamSynchronize(streamElim));
		// оба закончились, значит можно следующий col
	}
	// Освобождение память
	CUDA_CHECK(cudaFreeHost(h_pivot_pinned));
	CUDA_CHECK(cudaStreamDestroy(streamPivot));
	CUDA_CHECK(cudaStreamDestroy(streamUpdate));
	CUDA_CHECK(cudaStreamDestroy(streamElim));
}

int main() {
	setlocale(LC_ALL, "");
	auto start = chrono::high_resolution_clock::now();

	string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\11.11.bin)";
	string OutputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\output.bin)";
	bool isBis = false;

	size_t codeLength = 16200;  // n
	size_t infoLength = 3960;  // k
	double mCoeff = 2000.5;

	uint8_t* h_L = nullptr;
	uint8_t* d_L = nullptr;
	uint8_t* d_G_tmp = nullptr;
	uint8_t* d_base = nullptr;
	int* d_pivot = nullptr;
	size_t wordsCount = 0;


	cout << "Начинаем чтение" << endl;
	auto start1 = chrono::high_resolution_clock::now();
	if (!ReadCodeWords(InputFileName, codeLength, h_L, wordsCount, isBis, mCoeff))
		return 1;
	auto ms1 = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start1).count();
	cout << ms1 << endl;
	cout << "Закончили чтение " << endl;

	size_t bpr = bytesPerRow(codeLength);
	size_t L_bytes = wordsCount * bpr * sizeof(uint8_t);
	size_t G_bytes = codeLength * bpr * sizeof(uint8_t);

	uint8_t* G_tmp = createIdentityPacked(codeLength);

	CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
	CUDA_CHECK(cudaMemcpy(d_L, h_L, L_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc(&d_G_tmp, G_bytes));
	CUDA_CHECK(cudaMemcpy(d_G_tmp, G_tmp, G_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc(&d_base, bpr * sizeof(uint8_t)));
	CUDA_CHECK(cudaMalloc(&d_pivot, sizeof(int)));

	runIterativeEliminationGPU(
		wordsCount, codeLength, infoLength,
		d_L, d_G_tmp, d_pivot, d_base, (int)bpr);

	CUDA_CHECK(cudaMemcpy(G_tmp, d_G_tmp, G_bytes, cudaMemcpyDeviceToHost));
	WriteResultPacked(OutputFileName, G_tmp, codeLength, bpr, isBis);

	CUDA_CHECK(cudaFree(d_L));
	CUDA_CHECK(cudaFree(d_G_tmp));
	CUDA_CHECK(cudaFree(d_pivot));
	CUDA_CHECK(cudaFree(d_base));
	delete[] h_L;
	delete[] G_tmp;

	cout << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start).count() << endl;

	return 0;
}