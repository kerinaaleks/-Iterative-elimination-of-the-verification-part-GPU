#include <iostream>
#include <cstring>      // для memset и memcpy
#include <cstdint>
#include <iomanip>
#include <fstream>

using namespace std;

size_t codeLength = 4000; // длина кодового слова n
size_t infoLength = 4000; // число шагов (k)



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
	if (mat == nullptr) return;
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

bool ReadCodeWords(const string& filename, size_t codeLength, uint8_t**& L, size_t& wordsCount) {
	ifstream file(filename, ios::binary);
	if (!file.is_open()) {
		cout << "" << endl;
		return false;
	}

	file.seekg(0, ios::end);
	size_t fileSizeBytes = file.tellg();
	file.seekg(0, ios::beg);

	size_t totalBits = fileSizeBytes * 8;
	wordsCount = totalBits / codeLength;

	//
	if (wordsCount == 0) {
		cout << "" << endl;
		file.close();
		return false;
	}
	uint8_t* buffer = new uint8_t[fileSizeBytes];
	file.read(reinterpret_cast<char*>(buffer), fileSizeBytes);
	file.close();

	//
	L = new uint8_t*[wordsCount];
	for (size_t i = 0; i < wordsCount; i++) {
		L[i] = new uint8_t[codeLength];
		memset(L[i], 0, codeLength);
	}
	//
	size_t bitPos = 0;
	for (size_t w = 0; w < wordsCount; w++) {
		for (size_t b = 0; b < codeLength; b++) {
			size_t byteIndex = bitPos / 8;
			size_t bitIndex = bitPos % 8;

			L[w][b] = (buffer[byteIndex] >> bitIndex) & 1;
			bitPos++;
		}
	}
	delete[] buffer;
	return true;
}

void WriteResultToFile(
	const string& fileName,
	const uint8_t* h_matrix,
	size_t matrixRows,
	size_t frameLength,
	size_t codeLength,
	bool isBis) {
	ofstream out(fileName, ios::binary);
	if (!out.is_open()) {
		cerr << "\n";
		return;
	}

	if (isBis) {
		size_t totalBytes = matrixRows * codeLength;
		uint8_t* buf = new uint8_t[totalBytes]();
		size_t pos = 0;
		for (size_t row = 0; row < matrixRows; ++row) {
			const uint8_t* rowPtr = h_matrix + row * frameLength;
			for (size_t b = 0; b < codeLength; ++b) {
				if (GetBitHost(rowPtr, b))
					buf[pos] = 0xff;
				++pos;
			}
		}
		out.write(reinterpret_cast<char*>(buf), totalBytes);
		delete[] buf;
	}
	else {
		size_t totalBits = matrixRows * codeLength;
		size_t totalBytes = (totalBits + 7) / 8;
		uint8_t* buf = new uint8_t[totalBytes]();
		size_t outBitPos = 0;
		for (size_t row = 0; row < matrixRows; ++row) {
			const uint8_t* rowPtr = h_matrix + row * frameLength;
			for (size_t b = 0; b < codeLength; ++b) {
				if (GetBitHost(rowPtr, b))
			}
		}
	}

}

int main() {

	string InputFileName = R"(D:\Rubin\sessions\tmp_1783328386069\files\4.4.bin)";

	uint8_t** L = nullptr; // Матрица кодовых слов
	size_t wordsCount = 0;

	if (!ReadCodeWords(InputFileName, codeLength, L, wordsCount)) {
		return 1;
	}

	uint8_t** G_tmp = createIdentityMatrix(codeLength); // Накопленная матрица преобразований (результат работы алгоритма)

	uint8_t** G = createZeroMatrix(codeLength, codeLength); // Текущая матрица преобразования на одном шаге
	uint8_t** G_res = createZeroMatrix(codeLength, codeLength); // Временный результат умножения
	uint8_t** L_new = createZeroMatrix(wordsCount, codeLength); // Временная копия матрицы кодовых слов
	uint8_t* base = new uint8_t[codeLength];

	//printMatrix("L ", L, wordsCount, codeLength);
	//printMatrix("G_tmp", G_tmp, codeLength, codeLength);


	// ===================== Основной цикл (k шагов) =====================
	for (int col = 0; col < infoLength; col++) {

		cout << "========== Шаг col = " << col << " ==========" << endl;

		// 1. Ищем строку, у которой в столбце col стоит 1
		int pivot_row = -1;
		for (int row = 0; row < wordsCount; row++) {
			if (L[row][col] == 1) {
				pivot_row = (int)row;
				break;
			}
		}

		if (pivot_row == -1) {
			cout << "Базис не найден в столбце " << col << endl;
			continue;
		}

		// 2. Запоминаем базисный вектор
		memcpy(base, L[pivot_row], codeLength);

		//
		cout << "Базисная строка: " << pivot_row << " → ";
		for (int j = 0; j < codeLength; j++) {
			cout << (int)base[j] << " ";
		}
		cout << endl << endl;
		//

		// 3. Строим текущую матрицу G (единичная + хвост базиса)
		for (size_t i = 0; i < codeLength; i++) {
			memset(G[i], 0, codeLength);
			G[i][i] = 1;
		}
		// копируем хвост базисного вектора ( записываем в G хвост базиса в определенную строку)
		for (int j = col; j < codeLength; j++) {
			G[col][j] = base[j];
		}

		//printMatrix("Текущая G", G, codeLength, codeLength);

		// 4. Исключение столбца col (make_L_n)
		copyMatrix(L_new, L, wordsCount, codeLength);
		for (int row = 0; row < wordsCount; row++) {
			if (L[row][col] == 1) {
				// XOR хвоста с базисом
				for (size_t j = col + 1; j < codeLength; j++) {
					L_new[row][j] ^= base[j];
				}
			}
		}

		// обновляем L
		copyMatrix(L, L_new, wordsCount, codeLength);
		//printMatrix("L после исключения", L, wordsCount, codeLength);

		// 5. Умножение G_tmp = G_tmp * G  (над GF(2))

		for (size_t i = 0; i < codeLength; i++) {
			for (int j = 0; j < codeLength; j++) {
				uint8_t sum = 0;
				for (size_t k = 0; k < codeLength; k++) {
					sum ^= (G_tmp[i][k] & G[k][j]);   // умножение + XOR
				}
				G_res[i][j] = sum;
			}
		}

		// обновляем G_tmp
		copyMatrix(G_tmp, G_res, codeLength, codeLength);
		//printMatrix("G_tmp после умножения (накопление)", G_tmp, codeLength, codeLength);
	}

	cout << "=== ИТОГОВАЯ МАТРИЦА ПРЕОБРАЗОВАНИЯ G_res ===" << endl;
	printMatrix("G_res", G_tmp, codeLength, codeLength);

	freeMatrix(L, wordsCount);
	freeMatrix(G_tmp, codeLength);
	freeMatrix(G, codeLength);
	freeMatrix(G_res, codeLength);
	freeMatrix(L_new, wordsCount);
	delete[] base;

	return 0;
}