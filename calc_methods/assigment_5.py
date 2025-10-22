import numpy as np


def jacobi(A, b, x0=None, tol=1e-10, max_iter=1000):
    """Метод Якобі для розв'язання СЛАР"""
    n = len(b)

    # Початкове наближення
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    x_new = np.zeros(n)

    for iteration in range(max_iter):
        for i in range(n):
            # Обчислення нового значення x[i]
            sum_val = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_val) / A[i][i]

        # Перевірка збіжності
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Збіглося за {iteration + 1} ітерацій")
            return x_new

        x = x_new.copy()

    print(f"Досягнуто максимум ітерацій ({max_iter})")
    return x_new


# === ЗАДАТИ МАТРИЦЮ ТУТ ===
A = np.array([
    [10., -1., 2.],
    [-1., 11., -1.],
    [2., -1., 10.]
], dtype=float)

b = np.array([6., 25., -11.], dtype=float)
# ==========================

# Розв'язок
x = jacobi(A, b)

print("\nРозв'язок:")
for i, val in enumerate(x):
    print(f"x{i + 1} = {val:.6f}")

# Перевірка
print("\nПеревірка A*x:")
print(A @ x)
print("Вектор b:")
print(b)


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    """Метод Гауса-Зейделя (швидша збіжність)"""
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    for iteration in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            sum_val = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_val) / A[i][i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"Збіглося за {iteration + 1} ітерацій")
            return x

    print(f"Досягнуто максимум ітерацій ({max_iter})")
    return x


A = np.array([
    [10., -1., 2.],
    [-1., 11., -1.],
    [2., -1., 10.]
], dtype=float)

b = np.array([6., 25., -11.], dtype=float)

# Розв'язок
x = gauss_seidel(A, b)

print("\nРозв'язок:")
for i, val in enumerate(x):
    print(f"x{i+1} = {val:.6f}")

# Перевірка
print("\nПеревірка A*x:")
print(A @ x)
print("Вектор b:")
print(b)