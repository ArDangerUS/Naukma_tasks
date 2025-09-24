import numpy as np

def gauss_elimination(A, b):
    n = len(b)
    # Створюємо розширену матрицю
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    # Прямий хід
    for i in range(n):
        # Пошук головного елемента
        max_row = i + np.argmax(abs(Ab[i:, i]))
        if Ab[max_row, i] == 0:
            raise ValueError("Матриця вироджена або система не має єдиного розв'язку")

        # Міняємо рядки місцями
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        # Нормалізація та виключення змінних
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # Зворотний хід
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    return x

# Приклад використання
if __name__ == "__main__":
    A = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)

    x = gauss_elimination(A, b)
    print("Розв'язок системи:", x)
