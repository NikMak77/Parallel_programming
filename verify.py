import numpy as np

def read_matrix(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if 'Объем' in line or 'Время' in line or line.startswith('Объём'):
                break
            lines.append(line)

    if not lines:
        raise ValueError(f"Файл {filename} пустой или некорректный")

    try:
        n = int(lines[0])
    except ValueError:
        raise ValueError(f"Первая строка файла {filename} должна содержать размер матрицы (целое число), получено: {lines[0]}")

    matrix_lines = lines[1:1 + n]

    if len(matrix_lines) != n:
        raise ValueError(
            f"В файле {filename} недостаточно строк матрицы. "
            f"Ожидалось {n} строк после размера, найдено {len(matrix_lines)}"
        )

    try:
        matrix = np.array(
            [list(map(int, line.split())) for line in matrix_lines],
            dtype=np.int64
        )
    except ValueError as e:
        raise ValueError(f"Ошибка при парсинге чисел в матрице: {e}")

    if matrix.shape != (n, n):
        raise ValueError(
            f"Полученная матрица имеет размер {matrix.shape}, "
            f"ожидалась квадратная {n}×{n}"
        )

    return matrix, n


def main():
    try:
        A, nA = read_matrix('MatrixA.txt')
        B, nB = read_matrix('MatrixB.txt')
        C_got, nC = read_matrix('result.txt')

        if nA != nB or nA != nC:
            print("Ошибка: разные размеры матриц")
            print(f"MatrixA: {nA}×{nA}")
            print(f"MatrixB: {nB}×{nB}")
            print(f"result.txt: {nC}×{nC}")
            return

        print(f"Размеры матриц совпадают: {nA}×{nA}")

        C_expected = np.dot(A, B)

        if np.array_equal(C_expected, C_got):
            print("Верификация пройдена. Результаты полностью совпадают")
        else:
            print("Верификация НЕ пройдена!")
            print("\nРазличия найдены (первые 5 строк для примера):")
            diff = C_expected != C_got
            rows, cols = np.where(diff)
            for i in range(min(5, len(rows))):
                r, c = rows[i], cols[i]
                print(f"  [{r},{c}]  ожидалось {C_expected[r,c]}   получено {C_got[r,c]}")

    except Exception as e:
        print("Ошибка при выполнении верификации:")
        print(str(e))


if __name__ == "__main__":
    main()