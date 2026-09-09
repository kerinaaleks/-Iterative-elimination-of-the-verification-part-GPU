#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iostream>
#include <cstring>      // для memset и memcpy
#include <cstdint>
#include <iomanip>
#include <fstream>

using namespace std;

size_t codeLength = 2004; // длина кодового слова n
size_t infoLength = 2000; // число шагов (k)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(err) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while (0)

__global__ void eliminateColumnKernel(
	uint8_t* L,
	const uint8_t* base,
	int col,
	int wordsCount,
	int codeLength)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= wordsCount) return;

	uint8_t* rowPtr = L + row * codeLength; 
	// Если в опорном столбце стоит 1 - делаем XOR хвоста
	if (rowPtr[col] == 1){
		for (int j = col + 1; j < codeLength; j++) {
			rowPtr[j] ^= base[j];
		}
	}
}

__global__ void matMulKernel(
	const uint8_t* G_tmp,
	const uint8_t* G,
	uint8_t* G_res,
	int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= n || col >= n) return;

	uint8_t sum = 0;
	for (int k = 0; k < n; k++) {
		sum ^= (G_tmp[row * n + k] & G[k * n + col]);
	}
	G_res[row * n + col] = sum;
}

void matrixToFlat(uint8_t** src, uint8_t* dst, size_t n) {
	for (size_t i = 0; i < n; i++) {
		memcpy(dst + i * n, src[i], n);
	}
}

void flatToMatrix(uint8_t* src, uint8_t** dst, size_t n) {
	for (int i = 0; i < n; i++) {
		memcpy(dst[i], src + i * n, n);
	}
}

__device__ __forceinline__ bool GetBitDevice(const uint8_t* row, int bitIndex) {
	return (row[bitIndex / 8] >> (bitIndex % 8)) & 1;
}

inline bool GetBitHost(const uint8_t* row, size_t bitIndex) {
	return (row[bitIndex / 8] >> (bitIndex % 8)) & 1;
}

// Вывод матрицы
void printMatrix(const char* name, uint8_t** mat, size_t rows, size_t cols) {
	cout << name << "(" << rows << "x" << cols << "):" << endl;

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			cout << (int)mat[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void freeMatrix(uint8_t** mat, size_t rows) {
	if (!mat) return;
	for (size_t i = 0; i < rows; i++) {
		delete[] mat[i];
	}
	delete[] mat;
}

uint8_t** createIdentityMatrix(size_t n) {
	uint8_t** mat = new uint8_t*[n];
	for (size_t i = 0; i < n; i++) {
		mat[i] = new uint8_t[n];
		memset(mat[i], 0, n);
		mat[i][i] = 1;
	}
	return mat;
}

uint8_t** createZeroMatrix(size_t rows, size_t cols) {
	uint8_t** mat = new uint8_t*[rows];
	for (size_t i = 0; i < rows; i++) {
		mat[i] = new uint8_t[cols];
		memset(mat[i], 0, cols);
	}
	return mat;
}

void copyMatrix(uint8_t** dst, uint8_t** src, size_t rows, size_t cols) {
	for (size_t i = 0; i < rows; i++) {
		memcpy(dst[i], src[i], cols);
	}
}

// Чтение файла и преобразование в плоску матрицу
bool ReadCodeWords(const string& filename, size_t codeLength, uint8_t*& L, size_t& wordsCount) {
	ifstream file(filename, ios::binary);
	if (!file.is_open()) {
		cout << "Не удалось открыть файл" << endl;
		return false;
	}

	file.seekg(0, ios::end);
	size_t fileSizeBytes = file.tellg();
	file.seekg(0, ios::beg);

	size_t totalBits = fileSizeBytes * 8;
	wordsCount = totalBits / codeLength;

	//
	if (wordsCount == 0) {
		cout << "Недостаточно данных в файле" << endl;
		file.close();
		return false;
	}

	uint8_t* buffer = new uint8_t[fileSizeBytes];
	file.read(reinterpret_cast<char*>(buffer), fileSizeBytes);
	file.close();

	//
	uint8_t* L_flat = new uint8_t[wordsCount * codeLength];
	memset(L_flat, 0, wordsCount * codeLength);

	//
	size_t bitPos = 0;
	for (size_t w = 0; w < wordsCount; w++) {
		for (size_t b = 0; b < codeLength; b++) {
			size_t byteIndex = bitPos / 8;
			size_t bitIndex = bitPos % 8;

			L_flat[w * codeLength +b] = (buffer[byteIndex] >> bitIndex) & 1;
			bitPos++;
		}
	}
	delete[] buffer;
	L = L_flat;
	return true;
}

void WriteResultToFile(
	const string& fileName,
	uint8_t** G_tmp,
	size_t n,
	bool isBis) 
{
	ofstream out(fileName, ios::binary);
	if (!out.is_open()) {
		cerr << "Не удалось открыть файл для записи\n";
		return;
	}

	if (isBis) {
		size_t totalBytes = n * n;
		uint8_t* buf = new uint8_t[totalBytes];
		size_t pos = 0;
		for (size_t i = 0; i < n; ++i) {
			for (size_t j = 0; j < n; ++j) {
				buf[pos++] = G_tmp[i][j] ? 0xFF : 0x00;
			}
		}
		out.write(reinterpret_cast<char*>(buf), totalBytes);
		delete[] buf;
	}
	else {
		size_t totalBits = n * n;
		size_t totalBytes = (totalBits + 7) / 8;
		uint8_t* buf = new uint8_t[totalBytes]();
		size_t bitPos = 0;
		for (size_t i = 0; i < n; i++) {
			for (size_t j = 0; j < n; j++) {
				if (G_tmp[i][j]) {
					buf[bitPos / 8] |= (1u << (bitPos % 8));
				}
				bitPos++;
			}
		}
		out.write(reinterpret_cast<char*>(buf), totalBytes);
		delete[] buf;
	}
	out.close();
}

int main() {

	string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\4.4.bin)";
	string OutputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\output.bin)";
	bool isBis = false;

	uint8_t* h_L = nullptr; // плоская Матрица кодовых слов на хосте
	size_t wordsCount = 0;

	if (!ReadCodeWords(InputFileName, codeLength, h_L, wordsCount)) {
		return 1;
	}


	uint8_t* h_G_tmp = new uint8_t[codeLength * codeLength];
	uint8_t* h_G = new uint8_t[codeLength * codeLength];
	uint8_t* h_G_res = new uint8_t[codeLength * codeLength];

	uint8_t* d_G_tmp = nullptr;
	uint8_t* d_G = nullptr;
	uint8_t* d_G_res = nullptr;

	size_t G_bytes = codeLength * codeLength * sizeof(uint8_t);

	CUDA_CHECK(cudaMalloc(&d_G_tmp, G_bytes));
	CUDA_CHECK(cudaMalloc(&d_G, G_bytes));
	CUDA_CHECK(cudaMalloc(&d_G_res, G_bytes));




	uint8_t** G_tmp = createIdentityMatrix(codeLength); // Накопленная матрица преобразований (результат работы алгоритма)
	uint8_t** G = createZeroMatrix(codeLength, codeLength); // Текущая матрица преобразования на одном шаге
	uint8_t** G_res = createZeroMatrix(codeLength, codeLength); // Временный результат умножения
	//uint8_t** L_new = createZeroMatrix(wordsCount, codeLength); // Временная копия матрицы кодовых слов
	uint8_t* base = new uint8_t[codeLength];

	//printMatrix("L ", L, wordsCount, codeLength);
	//printMatrix("G_tmp", G_tmp, codeLength, codeLength);

	// Выделение памяти на ГПУ
	uint8_t* d_L = nullptr;
	uint8_t* d_base = nullptr;

	size_t L_bytes = wordsCount * codeLength * sizeof(uint8_t);

	CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
	CUDA_CHECK(cudaMalloc(&d_base, codeLength * sizeof(uint8_t)));
	// Копируем L на устройство 1 раз
	CUDA_CHECK(cudaMemcpy(d_L, h_L, L_bytes, cudaMemcpyHostToDevice));


	int threads = 256;
	int blocks = (wordsCount + threads - 1) / threads;

	// ===================== Основной цикл (k шагов) =====================
	for (int col = 0; col < infoLength; col++) {

		cout << "Шаг col = " << col << endl;

		// 1. Ищем строку, у которой в столбце col стоит 1
		int pivot_row = -1;
		for (int row = 0; row < wordsCount; row++) {
			if (h_L[row*codeLength +col] == 1) {
				pivot_row = (int)row;
				break;
			}
		}

		if (pivot_row == -1) {
			cout << "Базис не найден в столбце " << col << endl;
			continue;
		}

		// 2. Запоминаем базисный вектор
		memcpy(base, h_L + pivot_row * codeLength, codeLength);

		// 3. Строим текущую матрицу G (cpu)
		for (size_t i = 0; i < codeLength; i++) {
			memset(G[i], 0, codeLength);
			G[i][i] = 1;
		}
		for (int j = col; j < codeLength; j++) {
			G[col][j] = base[j];
		}

		// 4. ИСКЛЮЧЕНИЕ СТОЛБЦОВ НА ГПУ
		// Отправляем базис на гпу
		CUDA_CHECK(cudaMemcpy(d_base, base, codeLength, cudaMemcpyHostToDevice));

		eliminateColumnKernel << <blocks, threads >> > (d_L, d_base, col, (int)wordsCount, (int)codeLength);

		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// Забираем обновленную Л обратно на хост (нужно тк поиск пивотов идет на цпу)
		CUDA_CHECK(cudaMemcpy(h_L, d_L, L_bytes, cudaMemcpyDeviceToHost));

		//Умножение на ГПУ
		matrixToFlat(G_tmp, h_G_tmp, codeLength);
		matrixToFlat(G, h_G, codeLength);

		CUDA_CHECK(cudaMemcpy(d_G_tmp, h_G_tmp, G_bytes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_G, h_G, G_bytes, cudaMemcpyHostToDevice));

		dim3 block(16, 16);
		dim3 grid(
			(codeLength + block.x - 1) / block.x,
			(codeLength + block.y - 1) / block.y
		);

		matMulKernel << < grid, block >> > (d_G_tmp, d_G, d_G_res, (int)codeLength);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMemcpy(h_G_res, d_G_res, G_bytes, cudaMemcpyDeviceToHost));

		flatToMatrix(h_G_res, G_tmp, codeLength);

	
		//for (size_t i = 0; i < codeLength; i++) {
		//	for (int j = 0; j < codeLength; j++) {
		//		uint8_t sum = 0;
		//		for (size_t k = 0; k < codeLength; k++) {
		//			sum ^= (G_tmp[i][k] & G[k][j]);   // умножение + XOR
		//		}
		//		G_res[i][j] = sum;
		//	}
		//}

		//// обновляем G_tmp
		//copyMatrix(G_tmp, G_res, codeLength, codeLength);
		////printMatrix("G_tmp после умножения (накопление)", G_tmp, codeLength, codeLength);
	}

	if (isBis) {
		WriteResultToFile(OutputFileName, G_tmp, codeLength, true);
	}
	else {
		WriteResultToFile(OutputFileName, G_tmp, codeLength, false);
	}
	

	cout << "=== ИТОГОВАЯ МАТРИЦА ПРЕОБРАЗОВАНИЯ G_res ===" << endl;
	printMatrix("G_res", G_tmp, codeLength, codeLength);


	CUDA_CHECK(cudaFree(d_L));
	CUDA_CHECK(cudaFree(d_base));
	CUDA_CHECK(cudaFree(d_G_tmp));
	CUDA_CHECK(cudaFree(d_G));
	CUDA_CHECK(cudaFree(d_G_res));

	freeMatrix(G_tmp, codeLength);
	freeMatrix(G, codeLength);
	freeMatrix(G_res, codeLength);

	delete[] h_L;
	delete[] base;
	delete[] h_G_tmp;
	delete[] h_G;
	delete[] h_G_res;
		
	cout << "Закончили" << endl;
	return 0;
}