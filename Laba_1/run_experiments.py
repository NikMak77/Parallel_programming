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