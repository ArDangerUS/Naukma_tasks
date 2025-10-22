import numpy as np

# Вихідні дані
A = np.array([
    [25.00, 10.00, -7.50],
    [10.00, 29.00, -8.00],
    [-7.50, -8.00, 28.25]
], dtype=float)

b = np.array([60.00, 84.00, -80.00], dtype=float)

print("МЕТОД КВАДРАТНИХ КОРЕНІВ")

# Перевірка симетричності
is_symmetric = np.allclose(A, A.T)
print(f"\nМатриця симетрична: {is_symmetric}")

# Перевірка додатної визначеності
eigenvalues = np.linalg.eigvals(A)
is_positive_definite = np.all(eigenvalues > 0)
print(f"Матриця додатно визначена: {is_positive_definite}")
print(f"Власні значення: {eigenvalues}")

if not (is_symmetric and is_positive_definite):
    print("\nМетод Холецького не застосовний!")
    exit()

# Метод квадратних коренів
n = len(A)
S = np.eye(n)  # Верхня трикутна матриця
D = np.zeros(n)  # Діагональна матриця

# Розклад A = S^T * D * S
for i in range(n):
    # Обчислення D[i]
    sum_d = sum(S[k, i] ** 2 * D[k] for k in range(i))
    D[i] = A[i, i] - sum_d

    # Обчислення S[i, j]
    for j in range(i + 1, n):
        sum_s = sum(S[k, i] * D[k] * S[k, j] for k in range(i))
        S[i, j] = (A[i, j] - sum_s) / D[i]

print("\nМатриця S (верхня трикутна):")
print(S)

print("\nМатриця D (діагональна):")
print(np.diag(D))

# Перевірка розкладу
A_reconstructed = S.T @ np.diag(D) @ S
print("\nПеревірка A = S^T * D * S:")
print(A_reconstructed)
print(f"\nПохибка розкладу: {np.max(np.abs(A - A_reconstructed)):.2e}")

# Розв'язування системи
# 1. S^T * D * S * x = b
# 2. Нехай y = D * S * x, тоді S^T * y = b
# 3. Знаходимо y з S^T * y = b (пряма підстановка)
y = np.zeros(n)
for i in range(n):
    sum_val = sum(S.T[i, k] * y[k] for k in range(i))
    y[i] = (b[i] - sum_val) / S.T[i, i]

# 4. Нехай z = S * x, тоді D * z = y
z = y / D

# 5. Знаходимо x з S * x = z (зворотна підстановка)
x = np.zeros(n)
for i in range(n - 1, -1, -1):
    sum_val = sum(S[i, k] * x[k] for k in range(i + 1, n))
    x[i] = z[i] - sum_val

print("Рішення")
print(f"\nx₁ = {x[0]:.6f}")
print(f"x₂ = {x[1]:.6f}")
print(f"x₃ = {x[2]:.6f}")

# Перевірка
residual = A @ x - b
print("\nПеревірка A*x - b:")
print(residual)
print(f"\nНорма нев'язки: {np.linalg.norm(residual):.2e}")

x_numpy = np.linalg.solve(A, b)
print("\nПорівняння з numpy.linalg.solve:")
print(f"Різниця: {np.max(np.abs(x - x_numpy)):.2e}")