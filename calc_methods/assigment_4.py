import numpy as np

def thomas_algorithm(a, b, c, d):
    """
    Метод прогонки для тридіагональної системи
    a - піддіагональ
    b - головна діагональ
    c - наддіагональ
    d - права частина
    """
    n = len(d)
    alpha = np.zeros(n)
    beta = np.zeros(n)

    # Прямий хід
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * alpha[i - 1]
        if abs(denom) < 1e-10:
            raise ValueError("Ділення на нуль у методі прогонки!")
        alpha[i] = -c[i] / denom if i < n - 1 else 0
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom

    # Зворотний хід
    x = np.zeros(n)
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x, alpha, beta

# Вихідні дані
A = np.array([
    [4.080, 1.620, 0.000, 0.000, 0.000],
    [1.710, 4.400, 1.270, 0.000, 0.000],
    [0.000, 1.050, 4.780, 1.100, 0.000],
    [0.000, 0.000, 1.320, 4.810, 1.370],
    [0.000, 0.000, 0.000, 1.830, 4.620]
], dtype=float)

b_vec = np.array([1.620, 4.400, 1.050, 2.740, 9.240], dtype=float)


print("Метод прогонки")


# Перевірка тридіагональності
n = len(b_vec)
is_tridiagonal = True
for i in range(n):
    for j in range(n):
        if abs(i - j) > 1 and abs(A[i, j]) > 1e-10:
            is_tridiagonal = False
            break

print(f"\nМатриця тридіагональна: {is_tridiagonal}")

if not is_tridiagonal:
    print("ПОМИЛКА: Метод прогонки застосовний лише для тридіагональних матриць!")
    exit()

# Виділення діагоналей
a = np.zeros(n)  # піддіагональ
b = np.zeros(n)  # головна діагональ
c = np.zeros(n)  # наддіагональ

b[0] = A[0, 0]
c[0] = A[0, 1]

for i in range(1, n - 1):
    a[i] = A[i, i - 1]
    b[i] = A[i, i]
    c[i] = A[i, i + 1]

a[n - 1] = A[n - 1, n - 2]
b[n - 1] = A[n - 1, n - 1]

print("Діагоналі матриці:")
print(f"Піддіагональ (a): {a}")
print(f"Головна діагональ (b): {b}")
print(f"Наддіагональ (c): {c}")
print(f"Права частина (d): {b_vec}")

# Перевірка умови стійкості методу прогонки

print("перевірка умови стійкості")


stable = True
print("Умова стійкості: |b[i]| >= |a[i]| + |c[i]|")
print("Перевірка для кожного рядка:")

for i in range(n):
    left = abs(b[i])
    right = abs(a[i]) + abs(c[i])
    condition = left >= right
    print(f"Рядок {i + 1}: |{b[i]:.3f}| >= |{a[i]:.3f}| + |{c[i]:.3f}| => {left:.3f} >= {right:.3f} : {condition}")
    if not condition:
        stable = False

if stable:
    print("Умова стійкості виконується для всіх рядків")

else:
    print("\n✗ Умова стійкості НЕ ВИКОНУЄТЬСЯ")
    print("Метод прогонки може бути нестійким!")

# Розв'язування методом прогонки

print("Рішення")

x, alpha, beta = thomas_algorithm(a, b, c, b_vec)

print("Прогонні коефіцієнти α:")
for i, val in enumerate(alpha):
    print(f"α[{i}] = {val:.6f}")

print("Прогонні коефіцієнти β:")
for i, val in enumerate(beta):
    print(f"β[{i}] = {val:.6f}")


print("Розвязок системи")


for i, val in enumerate(x):
    print(f"x[{i + 1}] = {val:.6f}")

# Перевірка розв'язку

print("перевірка")


residual = A @ x - b_vec
print("\nНев'язка A*x - b:")
for i, val in enumerate(residual):
    print(f"r[{i + 1}] = {val:.2e}")

norm = np.linalg.norm(residual)
print(f"\nНорма нев'язки: {norm:.2e}")

# Порівняння з numpy
x_numpy = np.linalg.solve(A, b_vec)
print("\nПорівняння з numpy.linalg.solve:")
print(f"Максимальна різниця: {np.max(np.abs(x - x_numpy)):.2e}")

print("\nРозв'язок numpy:")
for i, val in enumerate(x_numpy):
    print(f"x[{i + 1}] = {val:.6f}")