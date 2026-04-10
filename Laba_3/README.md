# Отчёт: Умножение двух квадратных матриц на C++ + MPI
 
**Цель:** Модифицировать программу из ЛР №1 для параллельной работы по технологии MPI. Провести эксперименты для размеров 200–2000 и разного числа процессов (1, 2, 4, 8).

## 1. Описание решения

### Программа на C++
- **Вход:** два файла `matrixA_N.txt` и `matrixB_N.txt` (матрицы хранятся построчно, через пробел, без заголовка с N).
- **Выход:** файл `result_N.txt` (результирующая матрица), плюс в консоль:
  - Время выполнения умножения (в секундах, с точностью 4 знака).
  - Размер матрицы N × N.
  - Объём задачи (N³ операций).
- Алгоритм: классический тройной цикл (O(N³)), без внешних библиотек.
- Компиляция: `mpic++ -O2 -std=c++20 matrix_mul.cpp -o matrix_mul`.

### Автоматизированная верификация
Отдельный скрипт `verify.py` загружает матрицы через **NumPy**, вычисляет `A @ B` и сравнивает с результатом C++ (`np.allclose` с `atol=1e-5`).  
Если расхождение > 1e-5 — выводит максимальную ошибку.

### Эксперименты
Размеры: **200, 400, 800, 1200, 1600, 2000**.  
Для каждого размера:
1. Генерируются случайные матрицы
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
#include <mpi.h>

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
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 4) {
        if (rank == 0) {
            cerr << "Использование: mpirun -np <num> " << argv[0] << " <matrixA> <matrixB> <result>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    vector<vector<double>> full_A, full_B;
    int n = 0;

    if (rank == 0) {
        full_A = read_matrix(argv[1]);
        full_B = read_matrix(argv[2]);

        if (!is_square(full_A) || !is_square(full_B) || full_A.size() != full_B.size()) {
            cerr << "Ошибка: матрицы не квадратные или разных размеров!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n = static_cast<int>(full_A.size());
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n == 0) {
        MPI_Finalize();
        return 1;
    }

    vector<double> B_flat(n * n, 0.0);
    if (rank == 0) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                B_flat[i * n + j] = full_B[i][j];
    }
    MPI_Bcast(B_flat.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<vector<double>> B(n, vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            B[i][j] = B_flat[i * n + j];

    int rows_per_proc_base = n / world_size;
    int remainder = n % world_size;
    int local_rows = rows_per_proc_base + (rank < remainder ? 1 : 0);

    vector<int> sendcounts(world_size);
    vector<int> displs(world_size, 0);
    int current_displ = 0;
    for (int i = 0; i < world_size; ++i) {
        int proc_rows = rows_per_proc_base + (i < remainder ? 1 : 0);
        sendcounts[i] = proc_rows * n;
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }

    vector<double> A_flat(n * n, 0.0);
    if (rank == 0) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                A_flat[i * n + j] = full_A[i][j];
    }

    vector<double> A_local_flat(local_rows * n);
    MPI_Scatterv(A_flat.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
        A_local_flat.data(), local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<vector<double>> A_local(local_rows, vector<double>(n));
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < n; ++j)
            A_local[i][j] = A_local_flat[i * n + j];

    vector<vector<double>> C_local(local_rows, vector<double>(n, 0.0));

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = 0.0;
    if (rank == 0) start_time = MPI_Wtime();

    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C_local[i][j] += A_local[i][k] * B[k][j];

    vector<double> C_local_flat(local_rows * n, 0.0);
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < n; ++j)
            C_local_flat[i * n + j] = C_local[i][j];

    vector<int> recvcounts = sendcounts;
    vector<int> recvdispls = displs;
    vector<double> C_flat(n * n, 0.0);

    MPI_Gatherv(C_local_flat.data(), local_rows * n, MPI_DOUBLE,
        C_flat.data(), recvcounts.data(), recvdispls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double duration = MPI_Wtime() - start_time;

        vector<vector<double>> C(n, vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C[i][j] = C_flat[i * n + j];

        write_matrix(argv[3], C);

        long long volume = (long long)n * n * n;
        cout << "Время выполнения: " << fixed << setprecision(4) << duration << " секунд" << endl;
        cout << "Количество процессов: " << world_size << endl;
        cout << "Размер матрицы: " << n << " x " << n << endl;
        cout << "Объем задачи: " << volume << " операций" << endl;
    }

    MPI_Finalize();
    return 0;
}'''

### generate_matrices.py

'''
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
'''

### run_experiments.py

'''
import subprocess
import re
import os
import pandas as pd
import matplotlib.pyplot as plt

sizes = [200, 400, 800, 1200, 1600, 2000]
nps = [1, 2, 4, 8]
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

    for np_val in nps:
        try:
            proc = subprocess.run(["mpirun", "-np", str(np_val), "--oversubscribe", "./matrix_mul",
                                           a_file, b_file, res_file],
                                           capture_output=True, text=True, timeout=600)

            output = proc.stdout + proc.stderr

            time_match = re.search(r"Время выполнения: ([\d.]+) секунд", output)
            if time_match:
                t = float(time_match.group(1))
                vol = n * n * n
                results.append({"N": n, "NP": np_val, "Time_s": t, "Volume": vol})
                print(f"N={n:4d} | NP={np_val:2d} | Время: {t:8.4f} с")
            else:
                print(f"N={n} NP={np_val} — не удалось прочитать время")
                if output.strip():
                    print("Вывод программы:")
                    print(output.strip()[:800])

        except Exception as e:
            print(f"N={n} NP={np_val} — ошибка: {e}")

df = pd.DataFrame(results)
df.to_csv("results_mpi.csv", index=False)
print("\nРезультаты сохранены в results_mpi.csv")

# График
plt.figure(figsize=(12, 7))
for np_val in sorted(df["NP"].unique()):
    sub = df[df["NP"] == np_val]
    plt.plot(sub["N"], sub["Time_s"], marker='o', linewidth=2, label=f"{np_val} процессов")

plt.xlabel("Размер матрицы N")
plt.ylabel("Время выполнения (секунды)")
plt.title("Умножение матриц MPI (наивный алгоритм)")
plt.grid(True)
plt.legend()
plt.savefig("time_vs_n_mpi.png")
plt.show()
print("График сохранён: time_vs_n_mpi.png")
'''

### verify.py

'''
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
'''

## Инструкция по запуску

mpic++ -O2 -std=c++20 matrix_mul.cpp -o matrix_mul
python3 generate_matrices.py
python3 run_experiments.py
python3 verify.py matrixA_200.txt matrixB_200.txt result_200.txt

## Результаты экспериментов

### Таблица времени выполнения

Было проведено умножение квадратных матриц размера от 200 до 2000. Время замерялось c помощью стандартной оптимизацией

| Размер матрицы | NP=1(c)  | NP=2(c)  | NP=4(c)  | NP=8(c)  |   Кол-во операций  |
|----------------|----------|----------|----------|----------|--------------------|
| 200            | 0.0115   | 0.0072   | 0.0051   | 0.0041   |       8000000      |
| 400            | 0.0907   | 0.0601   | 0.0438   | 0.03     |      64000000      |
| 800            | 0.8441   | 0.7919   | 0.5105   | 0.4426   |      512000000     |
| 1200           | 4.0747   | 3.1254   | 2.3316   | 1.6044   |     1728000000     |
| 1600           | 13.3614  | 9.3693   | 6.9032   | 5.6047   |     4096000000     |
| 2000           | 26.7638  | 19.3342  | 13.7787  | 10.4707  |     8000000000     |

### График

![График](time_vs_n_mpi.png)

### Вывод

- Зона высокой эффективности (n = 200-400): Здесь объем вычислений еще мал (O(n³)), но коммуникационные затраты уже окупаются при малом числе процессов (NP=2,4). На 8 процессах эффективность падает до ~35%, так как дробление маленькой матрицы на 8 частей порождает избыточный трафик передачи данных.
- Зона стабильного ускорения (n = 800-2000): С ростом объема данных (от 512 млн до 8 млрд операций) ускорение перестало стремительно расти с увеличением числа ядер.
- При переходе с 4 на 8 ядер прирост производительности составляет лишь 30-40% (например, на n=2000: 1.94 -> 2.56).
- При 8 процессах мы упираемся либо в пропускную способность сети, либо в дисбаланс нагрузки между процессами.
- Архитектура тестового стенда накладывает серьезное ограничение на скорость обмена данными, что не позволяет достичь линейного ускорения.