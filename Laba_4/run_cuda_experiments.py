import subprocess
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sizes = [512, 1024, 1536, 2048]
block_sizes = [8, 16, 24, 32]
results = []

exe_name = "matrix_mul_cuda.exe"
if not os.path.exists(exe_name):
    print(f"Ошибка: исполняемый файл {exe_name} не найден.")
    print("Скомпилируйте его с помощью: nvcc -o matrix_mul_cuda.exe matrix_mul_cuda.cu")
    exit(1)

for n in sizes:
    a_file = f"matrixA_{n}.txt"
    b_file = f"matrixB_{n}.txt"
    res_file = f"result_{n}.txt"

    if not os.path.exists(a_file):
        mat = np.random.uniform(-5, 5, (n, n))
        np.savetxt(a_file, mat, fmt='%.6f')
        np.savetxt(b_file, mat, fmt='%.6f')
        print(f"Сгенерированы матрицы {n}x{n}")

    for bs in block_sizes:
        print(f"Запуск: N={n}, block={bs}x{bs}")
        try:
            proc = subprocess.run(
                [exe_name, a_file, b_file, res_file, str(bs)],
                capture_output=True, text=True, timeout=600
            )
            stdout = proc.stdout
            stderr = proc.stderr

            if stdout:
                print("--- STDOUT ---")
                print(stdout)
            if stderr:
                print("--- STDERR ---")
                print(stderr)

            time_match = re.search(r"([\d.eE+-]+)\s*секунд", stdout)
            if time_match:
                t = float(time_match.group(1))
                vol = n * n * n
                results.append({
                    "N": n,
                    "BlockSize": bs,
                    "Time_s": t,
                    "Volume": vol,
                    "GFLOPS": (2 * vol / t) / 1e9 if t > 0 else 0
                })
                print(f"  Время: {t:6e} с, GFLOPS: {results[-1]['GFLOPS']:.2f}")
            else:
                print(f"  Не удалось прочитать время. Вывод программы выше.")
        except Exception as e:
            print(f"  Ошибка: {e}")

if len(results) == 0:
    print("Нет ни одного успешного замера. Проверьте работу программы вручную.")
    exit(1)

df = pd.DataFrame(results)
df.to_csv("cuda_results.csv", index=False)
print("\nРезультаты сохранены в cuda_results.csv")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for bs in block_sizes:
    subset = df[df["BlockSize"] == bs]
    if not subset.empty:
        plt.plot(subset["N"], subset["Time_s"], marker='o', label=f'block={bs}')
plt.xlabel("Размер матрицы N")
plt.ylabel("Время выполнения (с)")
plt.title("Время умножения матриц (CUDA)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for bs in block_sizes:
    subset = df[df["BlockSize"] == bs]
    if not subset.empty:
        plt.plot(subset["N"], subset["GFLOPS"], marker='o', label=f'block={bs}')
plt.xlabel("Размер матрицы N")
plt.ylabel("GFLOPS")
plt.title("Производительность (CUDA)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cuda_performance.png")
plt.show()
print("Графики сохранены: cuda_performance.png")