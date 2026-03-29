import subprocess
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

sizes = [200, 400, 800, 1200, 1600, 2000]
threads_list = [1, 2, 4, 8]
results_file = "results.csv"

def run_experiments():
    results = []
    if not os.path.exists(results_file):
        print("Запуск экспериментов...")
        for n in sizes:
            for t in threads_list:
                print(f"  Размер {n} x {n}, потоков {t}")
                cmd = ["./matrix_mul.exe", f"matrixA_{n}.txt", f"matrixB_{n}.txt", f"result_{n}_{t}.txt", str(t)]
                try:
                    output = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    print(f"Ошибка: {e.output}")
                    continue
                for line in output.splitlines():
                    if "Время выполнения" in line:
                        time_str = line.split()[2]
                        time_val = float(time_str)
                        results.append([n, t, time_val])
                        break

        with open(results_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["size", "threads", "time"])
            writer.writerows(results)
        print(f"Результаты сохранены в {results_file}")
    else:
        print("Файл результатов уже существует, загружаем...")
        with open(results_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                results.append([int(row[0]), int(row[1]), float(row[2])])
    return results

def plot_results(results):
    data = {(size, threads): time for size, threads, time in results}
    
    plt.figure(figsize=(10, 6))
    for t in threads_list:
        times = []
        sizes_used = []
        for n in sizes:
            if (n, t) in data:
                sizes_used.append(n)
                times.append(data[(n, t)])
        plt.plot(sizes_used, times, marker='o', label=f"{t} потоков")
    plt.xlabel("Размер матрицы (n)")
    plt.ylabel("Время (секунды)")
    plt.title("Время умножения матриц")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.savefig("time_vs_size.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    for n in sizes:
        base_time = data.get((n, 1), None)
        if base_time is None:
            continue
        speedups = []
        threads_used = []
        for t in threads_list:
            if (n, t) in data:
                threads_used.append(t)
                speedups.append(base_time / data[(n, t)])
        plt.plot(threads_used, speedups, marker='s', label=f"n={n}")
    ideal_x = [1, max(threads_list)]
    ideal_y = [1, max(threads_list)]
    plt.plot(ideal_x, ideal_y, 'k--', label="Идеальное ускорение")
    plt.xlabel("Число потоков")
    plt.ylabel("Ускорение")
    plt.title("Ускорение от числа потоков")
    plt.legend()
    plt.grid(True)
    plt.savefig("speedup.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 6))
    for n in sizes:
        base_time = data.get((n, 1), None)
        if base_time is None:
            continue
        eff = []
        threads_used = []
        for t in threads_list:
            if (n, t) in data:
                threads_used.append(t)
                eff.append((base_time / data[(n, t)]) / t)
        plt.plot(threads_used, eff, marker='^', label=f"n={n}")
    plt.xlabel("Число потоков")
    plt.ylabel("Эффективность")
    plt.title("Эффективность распараллеливания")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig("efficiency.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    results = run_experiments()
    plot_results(results)
    print("Графики сохранены: time_vs_size.png, speedup.png, efficiency.png")