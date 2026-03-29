#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <omp.h>

using namespace std;

vector<vector<double>> read_matrix(const string& filename) {
    ifstream file(filename);
    vector<vector<double>> mat;
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        vector<double> row;
        double val;
        while (ss >> val) row.push_back(val);
        if (!row.empty()) mat.push_back(row);
    }
    return mat;
}

bool is_square(const vector<vector<double>>& m) {
    if (m.empty()) return false;
    size_t n = m.size();
    return all_of(m.begin(), m.end(), [n](const auto& r) { return r.size() == n; });
}

vector<vector<double>> multiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

void write_matrix(const string& filename, const vector<vector<double>>& mat) {
    ofstream file(filename);
    for (const auto& row : mat) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << fixed << setprecision(6) << row[i];
            if (i < row.size() - 1) file << " ";
        }
        file << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "Использование: " << argv[0] << " <matrixA> <matrixB> <result> <num_threads>" << endl;
        return 1;
    }

    int num_threads = atoi(argv[4]);
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);

    auto A = read_matrix(argv[1]);
    auto B = read_matrix(argv[2]);

    if (!is_square(A) || !is_square(B) || A.size() != B.size()) {
        cerr << "Ошибка: матрицы не квадратные или разных размеров!" << endl;
        return 1;
    }

    int n = A.size();

    auto start = chrono::high_resolution_clock::now();
    auto C = multiply(A, B);
    auto end = chrono::high_resolution_clock::now();

    double duration = chrono::duration<double>(end - start).count();

    write_matrix(argv[3], C);

    long long volume = (long long)n * n * n;

    cout << "Количество потоков: " << num_threads << endl;
    cout << "Время выполнения: " << fixed << setprecision(4) << duration << " секунд" << endl;
    cout << "Размер матрицы: " << n << " x " << n << endl;
    cout << "Объем задачи: " << volume << " операций" << endl;

    return 0;
}