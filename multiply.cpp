#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace std;

vector<vector<long long>> readMatrix(const string& filename, int& n) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла: " << filename << endl;
        exit(1);
    }
    file >> n;
    vector<vector<long long>> matrix(n, vector<long long>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file >> matrix[i][j];
        }
    }
    file.close();
    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<long long>>& matrix, int n) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка открытия файла для записи: " << filename << endl;
        exit(1);
    }
    file << n << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file << matrix[i][j];
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
    file.close();
}

vector<vector<long long>> multiplyMatrices(
    const vector<vector<long long>>& A,
    const vector<vector<long long>>& B,
    int n)
{
    vector<vector<long long>> C(n, vector<long long>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

int main() {
    string fileA = R"(C:\Users\Home\Desktop\Parallel_programming\Laba_1\matrixA.txt)";
    string fileB = R"(C:\Users\Home\Desktop\Parallel_programming\Laba_1\matrixB.txt)";
    string fileResult = "result.txt";

    int nA, nB;
    auto A = readMatrix(fileA, nA);
    auto B = readMatrix(fileB, nB);

    if (nA != nB) {
        cerr << "Матрицы должны быть квадратными и одинакового размера!" << endl;
        return 1;
    }
    int n = nA;

    auto start = chrono::high_resolution_clock::now();
    auto C = multiplyMatrices(A, B, n);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;
    long long operations = 2LL * n * n * n - (long long)n * n;

    writeMatrix(fileResult, C, n);

    ofstream file(fileResult, ios::app);
    if (file.is_open()) {
        file << "\nОбъем задачи: " << operations << " арифметических операций (умножения + сложения)\n";
        file << "Время выполнения: " << fixed << setprecision(6) << duration.count() << " сек\n";
        file.close();
    }
    else {
        cerr << "Не удалось дописать статистику в файл\n";
    }

    cout << "Размер матриц: " << n << "×" << n << endl;
    cout << "Время выполнения: " << fixed << setprecision(4) << duration.count() << " сек" << endl;
    cout << "Объём задачи: " << operations << " операций умножения/сложения" << endl;
    cout << "Результат сохранён в: " << fileResult << endl;

    return 0;
};