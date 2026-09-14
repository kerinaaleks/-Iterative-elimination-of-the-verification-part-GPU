#include "cuda_runtime.h" // cudaMalloc, cudaMemcry, ядра
#include "device_launch_parameters.h" // служебное Cuda
#include <device_functions.h> // device-функции Cuda

#include <iostream>
#include <fstream>
#include <cstring> // memset
#include <cstdint> // uint8_t, uint64_t
#include <locale.h> // setlocalle
#include <chrono> // замер времени 

using namespace std;

constexpr int THREADS_PER_BLOCK = 256; // Кол-во потоков в одном ядре

constexpr int BITS = 64; // в одном uint64_t храним 64 бита 

// Подсчет сколько раз по 64 бит нужно на строку (+ округляем до большего)
inline size_t wordsPerRow(size_t n) {
	return (n + BITS - 1) / BITS;
}//!!!!!!!!!!!!!!!!!!!!!!!!!!

__host__ __device__ inline int getBit(const uint64_t* row, int bit) {
	//row - массив uint64_t одной строкой
	//bit - номер бита
	return (int)((row[bit / BITS] >> (bit % BITS)) & 1ULL);
}//!!!!!!!!!!!!!!!!!!!!!!!!!!

__host__ __device__ inline void setBit(uint64_t* row, int bit, int val) {
	const uint64_t mask = 1ULL << (bit%BITS);
	if (val) row[bit / BITS] |= mask;
	else row[bit / BITS] &= ~mask;
}//!!!!!!!!!!!!!!!!!!!!!!!!!!


// Макрос проверки CUDA
//если есть ошибка, выводит текст, файл, номер строки и завершает программу
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(err) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while (0)

// Поиск опорной строки 
__global__ void findPivotKernel(
	const uint64_t* L, // вся матрица кодовых слов на ГПУ
	int col, // текущий столбец
	int wordsCount, //число строк М
	int wpr, // слов uint64_t на одну строку
	int* pivotRow) // указатель на int на ГПУ (сюда пишется номер строки)
{
	int row = blockIdx.x *blockDim.x + threadIdx.x; // глобальный поток = номер строки 
	if (row >= wordsCount) return; // лшние потоки( если м не делится на 256) выходят

	const uint64_t* rowPtr = L + (size_t)row * wpr; // указатель на начало строки rowв в плоском массиве L
	if (getBit(rowPtr, col)) { // если в столбце col единица
		atomicMin(pivotRow, row); // пытаемся записать как мин( проверяем с предыдущим мин
	}
}

//xor хвоста с базисом base
__device__ void xorTailFast(
	uint64_t* row,
	const uint64_t* base, // базисная строка( на которую умножаем)
	int col, // с какогшо бита строки начинается хвост
	int n) // длина
{
	int start = col + 1;
	if (start >= n) return; // возвращаем если хвоста нет

	int startWord = start / 64; // с какого uint64_t начинаем
	int startOff = start % 64; // смешение анутри слова
	int nWords = n / 64; // сколько полных слов в длине n

	// если хвост начинается не с границы слова
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

// Обновление матрицы преобразования 
// если G_tmp[row][col] == 1, XOR хвоста строки row с base
__global__ void updateGTmpKernel(
	uint64_t* G_tmp, // накопленная матрица на ГПУ
	const uint64_t* base, // опорная трока (копия)
	int col, // текущий шаг
	int n, // размер 
	int wpr) // слов на строку(по 64)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x; // одна строка G_tmp на поток
	if (row >= n) return;

	uint64_t* rowPtr = G_tmp + (size_t)row * wpr;
	if (!getBit(rowPtr, col)) return; // не меняем строку

	xorTailFast(rowPtr, base, col, n);
	//!!!!!!!!!!!!!!!!!!!!!!!!!!
}

// Исключение столбца L
// та же идея xor, но для всех кодовых слов
// если L[row][col] == 1, XOR хвоста L[row] с base
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

// Создает матрицу на ЦПУ I, размера n х n
uint64_t* createIdentityPacked(size_t n) {
	size_t wpr = wordsPerRow(n);
	uint64_t* mat = new uint64_t[n * wpr]; // выделяем слова
	memset(mat, 0, n * wpr * sizeof(uint64_t)); // обнуляем матрицу
	for (size_t i = 0; i < n; i++) {
		setBit(mat + i * wpr, (int)i, 1); // ставим mat[i][i] = 1
	}
	return mat;
}

// Чтение файла
bool ReadCodeWords(
	const string& filename, // путь к файлу
	size_t codeLength, // n
	uint64_t*& L, // ссылка на указатель( ф-ция сама выделит массив и вернет через L)
	size_t& wordsCount, // М
	bool isBis,
	double mCoeff)
{
	ifstream file(filename, ios::binary); // Открываем файл
	if (!file.is_open()) {
		cout << "Не удалось открыть файл" << endl;
		return false;
	}

	file.seekg(0, ios::end);
	size_t fileSizeBytes = (size_t)file.tellg(); // узнаем разме файла в байтах
	file.seekg(0, ios::beg);

	size_t fileBits = isBis ? fileSizeBytes : fileSizeBytes * 8; // сколько бит есть в файле

	size_t wantBits = (size_t)(mCoeff * (double)codeLength * (double)codeLength); // сколько бит хотим взять
	wantBits = (wantBits / codeLength) * codeLength; // только целые кодовые слова

	size_t totalBits = (wantBits < fileBits) ? wantBits : fileBits;
	totalBits = (totalBits / codeLength) * codeLength;

	wordsCount = totalBits / codeLength; // сколько кодовых слов

	if (wordsCount == 0) {
		cout << "Недостаточно данных в файле" << endl;
		file.close();
		return false;
	}

	//cout << "Кодовых слов: " << wordsCount << ", n = " << codeLength << endl;

	size_t bytesToRead = isBis ? totalBits : (totalBits + 7) / 8;
	if (bytesToRead > fileSizeBytes)
		bytesToRead = fileSizeBytes;

	uint8_t* buffer = new uint8_t[bytesToRead];
	file.read(reinterpret_cast<char*>(buffer), bytesToRead); // читаем весь файл в буффер
	file.close();

	const size_t wpr = wordsPerRow(codeLength);
	L = new uint64_t[wordsCount * wpr]; // выделяем L на wordsCount * wpr слов
	memset(L, 0, wordsCount * wpr * sizeof(uint64_t)); // и обнуляем

	if (isBis) { // если формат Bis
		size_t pos = 0;
		for (size_t w = 0; w < wordsCount; w++) {
			uint64_t* row = L + w * wpr;
			for (size_t b = 0; b < codeLength; b++) {
				if (buffer[pos] != 0)
					setBit(row, (int)b, 1);
				pos++;
			}
		}
	}
	else { // если формат Bin
		size_t bitPos = 0;
		for (size_t w = 0; w < wordsCount; w++) {
			uint64_t* row = L + w * wpr;
			for (size_t b = 0; b < codeLength; b++) {
				size_t byteIndex = bitPos / 8;
				size_t bitIndex = bitPos % 8;
				if (byteIndex < bytesToRead) {
					int bit = (buffer[byteIndex] >> bitIndex) & 1;
					if (bit) setBit(row, (int)b, 1);
				}
				bitPos++;
			}
		}
	}

	delete[] buffer;
	return true;
}

// Запись результата в файл 
void WriteResultPacked(
	const string& path,
	const uint64_t* G,
	size_t n,
	size_t wpr,
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
			const uint64_t* row = G + i * wpr;
			for (size_t j = 0; j < n; j++) {
				buf[pos++] = getBit(row, (int)j) ? 0xFF : 0x00;
			}
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
			const uint64_t* row = G + i * wpr;
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

// Основной алгоритм
void runIterativeEliminationGPU(
	size_t wordsCount, // М
	size_t codeLength, // n
	size_t infoLength, // k
	uint64_t* d_L, // кодовые слова
	uint64_t* d_G_tmp, // накопленные преобразования
	int* d_pivot, // ячейка под номером pivot
	uint64_t* d_base, // буфер под копию опорной строки
	int wpr) // слов (по 64) на строку
{
	int blocks = (int)((wordsCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK); // сколько блоков, чтобы покрыть М строк
	int gBlocks = (int)((codeLength + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK); // сколько блоков, чтобы покрыть n строк G_tmp

	for (int col = 0; col < (int)infoLength; col++) {
		int h_pivot = (int)wordsCount; // маркер: не найдено
		CUDA_CHECK(cudaMemcpy(d_pivot, &h_pivot, sizeof(int), cudaMemcpyHostToDevice)); // делаем копию

		findPivotKernel << <blocks, THREADS_PER_BLOCK >> > ( // поиск лид.ед-цы
			d_L, col, (int)wordsCount, wpr, d_pivot);
		CUDA_CHECK(cudaDeviceSynchronize()); // ждем ГПУ

		CUDA_CHECK(cudaMemcpy(&h_pivot, d_pivot, sizeof(int), cudaMemcpyDeviceToHost)); // копируем на ЦПУ
		if (h_pivot >= (int)wordsCount) continue; //если столбца нет, продолжаем

		CUDA_CHECK(cudaMemcpy(d_base, d_L + (size_t)h_pivot * wpr, (size_t)wpr * sizeof(uint64_t),
			cudaMemcpyDeviceToDevice)); // копия строки d_L[h_pivot] -> d_base (чтобы base не портился при exclude)

		updateGTmpKernel << <gBlocks, THREADS_PER_BLOCK >> > (// обновляем G_tmp
			d_G_tmp, d_base, col, (int)codeLength, wpr);

		eliminateColumnKernel << <blocks, THREADS_PER_BLOCK >> > ( // обновляем L
			d_L, d_base, col, (int)wordsCount, (int)codeLength, wpr);

		CUDA_CHECK(cudaDeviceSynchronize()); // ждем гпу
	}
}

int main() {
	setlocale(LC_ALL, "");

	auto start = chrono::high_resolution_clock::now();

	string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\4.4.bin)";
	string OutputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\output.bin)";
	bool isBis = false;
	size_t codeLength = 1920; // n - длина кодового слова
	size_t infoLength = 1280; // k - сколько столбцов/шагов исключения
	double mCoeff = 28.0;

	// kMax

	uint64_t* h_L = nullptr;
	uint64_t* d_L = nullptr;
	uint64_t* d_G_tmp = nullptr;
	uint64_t* d_base = nullptr;
	int* d_pivot = nullptr;

	size_t wordsCount = 0;
	cout << "Начинаем чтение" << endl;
	auto start1 = chrono::high_resolution_clock::now();
	if (!ReadCodeWords(InputFileName, codeLength, h_L, wordsCount, isBis, mCoeff)) {; /////////////ЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪ
		return 1;
	}
	auto ms1 = chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start1).count();
	cout << ms1 << endl;
	cout << "Закончили чтение " << endl;

	size_t wpr = wordsPerRow(codeLength);
	size_t L_words = wordsCount * wpr;

	size_t L_bytes = L_words * sizeof(uint64_t);
	size_t G_bytes = codeLength * wpr * sizeof(uint64_t);
	cout << "Начинаем создавать единичную матрицу" << endl;
	uint64_t* G_tmp = createIdentityPacked(codeLength); /////////////ЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪЪ
	cout << "Закончили создавать единичнуюю матрицу" << endl;

	// Память ГПУ
	cout << "Начинаем выделять память на гпу" << endl;
	CUDA_CHECK(cudaMalloc(&d_L, L_bytes));
	CUDA_CHECK(cudaMemcpy(d_L, h_L, L_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc(&d_G_tmp, G_bytes));
	CUDA_CHECK(cudaMemcpy(d_G_tmp, G_tmp, G_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc(&d_base, wpr * sizeof(uint64_t)));
	CUDA_CHECK(cudaMalloc(&d_pivot, sizeof(int)));
	cout << "Закончили выделять память на гпу, начинаем основную функцию" << endl;

	runIterativeEliminationGPU(
		wordsCount,
		codeLength,
		infoLength,
		d_L,
		d_G_tmp,
		d_pivot,
		d_base,
		wpr
	);

	cout << "Закончили основную функцию" << endl;

	CUDA_CHECK(cudaMemcpy(G_tmp, d_G_tmp, G_bytes, cudaMemcpyDeviceToHost));
	WriteResultPacked(OutputFileName, G_tmp, codeLength, wpr, isBis);
	cout << "Закончили запись в файл" << endl;

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