# Отчёт: Умножение двух квадратных матриц на C++
 
**Цель:** Написать программу на C++ для умножения матриц, обеспечить ввод/вывод через файлы, измерение времени и объёма задачи, автоматизированную верификацию на Python + NumPy, провести эксперименты для размеров 200–2000.

## 1. Описание решения

### Программа на C++
- **Вход:** два файла `matrixA_N.txt` и `matrixB_N.txt` (матрицы хранятся построчно, через пробел, без заголовка с N).
- **Выход:** файл `result_N.txt` (результирующая матрица), плюс в консоль:
  - Время выполнения умножения (в секундах, с точностью 4 знака).
  - Размер матрицы N × N.
  - Объём задачи (N³ операций).
- Алгоритм: классический тройной цикл (O(N³)), без внешних библиотек.
- Компиляция: `g++ -O2 -std=c++17 matrix_mul.cpp -o matrix_mul`.

### Автоматизированная верификация
Отдельный скрипт `verify.py` загружает матрицы через **NumPy**, вычисляет `A @ B` и сравнивает с результатом C++ (`np.allclose` с `atol=1e-5`).  
Если расхождение > 1e-5 — выводит максимальную ошибку.

### Эксперименты
Размеры: **200, 400, 800, 1200, 1600, 2000**.  
Для каждого размера:
1. Генерируются случайные матрицы (диапазон [-5, 5]).
2. Запускается `./matrix_mul`.
3. Замеряется время.
4. Выполняется верификация.

Скрипт `run_experiments.py` полностью автоматизирует процесс и сохраняет `results.csv` + график `time_vs_n.png`.

## 2. Файлы проекта

### matrix_mul.cpp
```
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>

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
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C[i][j] += A[i][k] * B[k][j];
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
    if (argc != 4) {
        cerr << "Использование: " << argv[0] << " <matrixA> <matrixB> <result>" << endl;
        return 1;
    }

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

    cout << "Время выполнения: " << fixed << setprecision(4) << duration << " секунд" << endl;
    cout << "Размер матрицы: " << n << " x " << n << endl;
    cout << "Объем задачи: " << volume << " операций" << endl;

    return 0;
}
```

### generate_matrices.py

```
import numpy as np
import os

sizes = [200, 400, 800, 1200, 1600, 2000]

for n in sizes:
    a_file = f"matrixA_{n}.txt"
    b_file = f"matrixB_{n}.txt"
    if not os.path.exists(a_file):
        mat = np.random.uniform(-5, 5, (n, n))
        np.savetxt(a_file, mat, fmt='%.6f')
        np.savetxt(b_file, mat, fmt='%.6f')
        print(f"Сгенерированы матрицы {n}x{n}")
```

### run_experiments.py

```
import subprocess
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

sizes = [200, 400, 800, 1200, 1600, 2000]
results = []

for n in sizes:
    a_file = f"matrixA_{n}.txt"
    b_file = f"matrixB_{n}.txt"
    res_file = f"result_{n}.txt"

    if not os.path.exists(a_file):
        import numpy as np
        mat = np.random.uniform(-5, 5, (n, n))
        np.savetxt(a_file, mat, fmt='%.6f')
        np.savetxt(b_file, mat, fmt='%.6f')

    try:
        proc = subprocess.run(["./matrix_mul", a_file, b_file, res_file],
                              capture_output=True, text=True, timeout=600)
        output = proc.stdout

        time_match = re.search(r"Время выполнения: ([\d.]+) секунд", output)
        if time_match:
            t = float(time_match.group(1))
            vol = n * n * n
            results.append({"N": n, "Time_s": t, "Volume": vol})
            print(f"N={n:4d} | Время: {t:8.4f} с | Объём: {vol:,} оп.")
        else:
            print(f"N={n} — не удалось прочитать время")
    except Exception as e:
        print(f"N={n} — ошибка: {e}")

df = pd.DataFrame(results)
df.to_csv("results.csv", index=False)
print("\nРезультаты сохранены в results.csv")

plt.figure(figsize=(12, 6))
plt.plot(df["N"], df["Time_s"], marker='o', linewidth=2)
plt.xlabel("Размер матрицы N")
plt.ylabel("Время выполнения (секунды)")
plt.title("Время умножения матриц (C++, наивный алгоритм)")
plt.grid(True)
plt.savefig("time_vs_n.png")
plt.show()
print("График сохранён: time_vs_n.png")
```

### verify.py

```
import numpy as np
import sys

if len(sys.argv) != 4:
    print("Использование: python verify.py matrixA.txt matrixB.txt result.txt")
    sys.exit(1)

A = np.loadtxt(sys.argv[1])
B = np.loadtxt(sys.argv[2])
C_cpp = np.loadtxt(sys.argv[3])

C_py = A @ B

if np.allclose(C_cpp, C_py, atol=1e-5):
    print("Верификация успешна! Результаты совпадают.")
else:
    max_diff = np.max(np.abs(C_cpp - C_py))
    print("Верификация не пройдена!")
    print(f"Максимальное расхождение: {max_diff}")
```

## Инструкция по запуску

g++ -O2 -std=c++17 matrix_mul.cpp -o matrix_mul
python generate_matrices.py
python run_experiments.py
python verify.py matrixA_200.txt matrixB_200.txt result_200.txt

## Результаты экспериментов

### Таблица времени выполнения

Было проведено умножение квадратных матриц размера от 200 до 2000. Время замерялось c помощью стандартной оптимизацией

| Размер матрицы | -O2 (с)  |   Кол-во операций  |
|----------------|----------|--------------------|
| 200            | 0.0096   |       8000000      |
| 400            | 0.0624   |      64000000      |
| 800            | 0.6784   |      512000000     |
| 1200           | 4.5509   |     1728000000     |
| 1600           | 11.4956  |     4096000000     |
| 2000           | 31.9406  |     8000000000     |

### График

![График](time_vs_n.png)

### Вывод

- С ростом размера матрицы время растёт кубически (O(n³))
- Предоставлены график и таблица с результатами