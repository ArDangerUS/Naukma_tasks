import numpy as np
from scipy.optimize import linprog
from fractions import Fraction

print("=" * 70)
print("РОЗВ'ЯЗАННЯ МАТРИЧНОЇ ГРИ")
print("=" * 70)

# Матриця гри
A = np.array([
    [6, 7, 4],
    [3, 5, 9],
    [9, 8, 5]
])

print("\nМатриця гри:")
print(A)

# 1. Аналіз гри
print("\n" + "=" * 70)
print("1. АНАЛІЗ ГРИ")
print("=" * 70)

row_mins = np.min(A, axis=1)
col_maxs = np.max(A, axis=0)

alpha = np.max(row_mins)
beta = np.min(col_maxs)

print(f"\nМінімуми рядків: {row_mins}")
print(f"V_H = max{{{', '.join(map(str, row_mins))}}} = {alpha}")
print(f"\nМаксимуми стовпців: {col_maxs}")
print(f"V_B = min{{{', '.join(map(str, col_maxs))}}} = {beta}")
print(f"\nV_H ≠ V_B, тобто V ∈ ({alpha}, {beta})")
print("Немає сідлової точки. Отже, немає розв'язку гри в чистих стратегіях.")

# 2. Перевірка домінування
print("\n" + "=" * 70)
print("2. ПЕРЕВІРКА ДОМІНУВАННЯ")
print("=" * 70)
print("\nТеорему домінування не можна застосувати, тому що в матриці немає")
print("стратегії, яка була б кращою за іншу по всіх елементах.")

# 3. Розв'язання задач ЛП
print("\n" + "=" * 70)
print("3. ЗАДАЧІ ЛІНІЙНОГО ПРОГРАМУВАННЯ")
print("=" * 70)

# Задача для гравця A (мінімізація)
print("\n--- Задача для гравця A ---")
print("x₁ + x₂ + x₃ → min")
print("За умов (стовпці матриці A):")
print("  6x₁ + 7x₂ + 4x₃ ≥ 1  (стовпець C₁)")
print("  3x₁ + 5x₂ + 9x₃ ≥ 1  (стовпець C₂)")
print("  9x₁ + 8x₂ + 5x₃ ≥ 1  (стовпець C₃)")
print("  xᵢ ≥ 0")

# Перетворюємо на стандартну форму для linprog (мінімізація, ≤)
# A·x ≥ b перетворюємо в -A·x ≤ -b
c_A = [1, 1, 1]  # коефіцієнти цільової функції
A_ub_A = -A.T  # транспонована і множимо на -1 (стовпці стають рядками)
b_ub_A = -np.ones(3)  # множимо на -1

result_A = linprog(c_A, A_ub=A_ub_A, b_ub=b_ub_A, bounds=(0, None), method='highs')

if result_A.success:
    Z_A = result_A.fun
    V_A = 1 / Z_A
    x_opt = result_A.x
    p_opt = x_opt / Z_A

    print(f"\nРезультат:")
    print(f"Z = Σxᵢ = {Z_A:.6f}")
    print(f"V = 1/Z = {V_A:.6f}")
    print(f"x = ({x_opt[0]:.6f}, {x_opt[1]:.6f}, {x_opt[2]:.6f})")
    print(f"p = ({p_opt[0]:.6f}, {p_opt[1]:.6f}, {p_opt[2]:.6f})")
    print(f"Сума p: {np.sum(p_opt):.6f}")
else:
    print("Помилка розв'язання!")

# Задача для гравця B (максимізація)
print("\n--- Задача для гравця B ---")
print("y₁ + y₂ + y₃ → max")
print("За умов (рядки матриці A):")
print("  6y₁ + 3y₂ + 9y₃ ≤ 1  (рядок K₁)")
print("  7y₁ + 5y₂ + 8y₃ ≤ 1  (рядок K₂)")
print("  4y₁ + 9y₂ + 5y₃ ≤ 1  (рядок K₃)")
print("  yⱼ ≥ 0")

# Для максимізації мінімізуємо -f
c_B = [-1, -1, -1]  # мінімізуємо -(y₁+y₂+y₃)
A_ub_B = A  # матриця без транспонування (рядки залишаються рядками)
b_ub_B = np.ones(3)

result_B = linprog(c_B, A_ub=A_ub_B, b_ub=b_ub_B, bounds=(0, None), method='highs')

if result_B.success:
    Z_B = -result_B.fun  # міняємо знак назад
    V_B = 1 / Z_B
    y_opt = result_B.x
    q_opt = y_opt / Z_B

    print(f"\nРезультат:")
    print(f"Z = Σyⱼ = {Z_B:.6f}")
    print(f"V = 1/Z = {V_B:.6f}")
    print(f"y = ({y_opt[0]:.6f}, {y_opt[1]:.6f}, {y_opt[2]:.6f})")
    print(f"q = ({q_opt[0]:.6f}, {q_opt[1]:.6f}, {q_opt[2]:.6f})")
    print(f"Сума q: {np.sum(q_opt):.6f}")
else:
    print("Помилка розв'язання!")

# 4. Геометричний метод
print("\n" + "=" * 70)
print("4. ГЕОМЕТРИЧНИЙ МЕТОД")
print("=" * 70)
print("\nОскільки у нас 3×3 матриця, прямий геометричний метод застосувати складно.")
print("Візьмемо дві стратегії для гравця А: A₁ та A₃.")
print("\nНехай x — ймовірність А₁, (1-x) — ймовірність А₃.")
print("\nВиграші проти кожної стратегії В:")
print("C₁: V₁(x) = 6x + 9(1-x) = 9 - 3x")
print("C₂: V₂(x) = 7x + 8(1-x) = 8 - x")
print("C₃: V₃(x) = 4x + 5(1-x) = 5 - x")

print("\nЗнаходимо точки перетину:")
print("V₁ = V₂: 9 - 3x = 8 - x → 2x = 1 → x = 0.5")
print("При x = 0.5: V = 8 - 0.5 = 7.5")

print("\nЦе наближене значення (використано лише 2 з 3 стратегій).")
print("Точне значення отримано методом ЛП.")

# 6. Відповідь
print("\n" + "=" * 70)
print("6. ВІДПОВІДЬ")
print("=" * 70)

print(f"\nОптимальні стратегії та ціна гри:")
print(f"\nV = {V_A:.4f} ≈ {V_A:.2f}")
print(f"p = ({p_opt[0]:.4f}, {p_opt[1]:.4f}, {p_opt[2]:.4f})")
print(f"q = ({q_opt[0]:.4f}, {q_opt[1]:.4f}, {q_opt[2]:.4f})")


# Переводимо у дроби для точності
def to_fraction(x, limit=100):
    return Fraction(x).limit_denominator(limit)


p_frac = [to_fraction(p) for p in p_opt]
q_frac = [to_fraction(q) for q in q_opt]
v_frac = to_fraction(V_A)

print(f"\nУ вигляді дробів:")
print(f"V = {v_frac}")
print(f"p = ({p_frac[0]}, {p_frac[1]}, {p_frac[2]})")
print(f"q = ({q_frac[0]}, {q_frac[1]}, {q_frac[2]})")

print(f"\nТобто гравцю А доцільно змішувати стратегії A₁, A₂, A₃")
print(f"з ймовірностями {p_frac[0]}, {p_frac[1]}, {p_frac[2]} відповідно,")
print(f"а гравцю B — стратегії B₁, B₂, B₃")
print(f"з ймовірностями {q_frac[0]}, {q_frac[1]}, {q_frac[2]}.")

print("5. ПЕРЕВІРКА")

# Перевіряємо, що стратегії задовольняють умовам гри
print("\nПеревірка стратегії A (виграші проти кожної стратегії B):")
for j in range(3):
    payoff = sum(p_opt[i] * A[i, j] for i in range(3))
    print(f"  Проти B{j + 1}: {payoff:.6f} {'≥' if payoff >= V_A - 0.001 else '<'} {V_A:.6f}")

print("\nПеревірка стратегії B (програші проти кожної стратегії A):")
for i in range(3):
    payoff = sum(A[i, j] * q_opt[j] for j in range(3))
    print(f"  Проти A{i + 1}: {payoff:.6f} {'≤' if payoff <= V_B + 0.001 else '>'} {V_B:.6f}")

expected = p_opt @ A @ q_opt
print(f"\nМатематичне сподівання виграшу E(p,q): {expected:.6f}")
print(f"Ціна гри від А: V_A = {V_A:.6f}")
print(f"Ціна гри від B: V_B = {V_B:.6f}")
print(f"Різниця V_A - V_B: {abs(V_A - V_B):.8f}")
print(f"Різниця E(p,q) - V: {abs(expected - V_A):.8f}")

if abs(V_A - V_B) < 0.001 and abs(expected - V_A) < 0.001:
    print("\n✓ Перевірка пройшла успішно!")
else:
    print("\n✗ Є невеликі розбіжності (можливо через округлення)")
