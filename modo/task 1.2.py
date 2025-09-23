import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 12

print("Система обмежень:")
print("x₁ + 5x₂ ≥ 5")
print("5x₁ + x₂ ≥ 5")
print("x₁ + x₂ ≤ 6")
print("3x₁ - 4x₂ ≤ 12")
print("-4x₁ + 3x₂ ≤ 12")
print("x₁, x₂ ≥ 0")
print("Функція цілі: Z = 4x₁ - x₂ → max")


plt.figure(figsize=(12, 10))

# Діапазон значень
x1_range = np.linspace(-1, 8, 1000)


# Функції для обмежень
def constraint_1(x1): return (5 - x1) / 5  # x₁ + 5x₂ ≥ 5 → x₂ ≥ (5-x₁)/5


def constraint_2(x1): return 5 - 5 * x1  # 5x₁ + x₂ ≥ 5 → x₂ ≥ 5-5x₁


def constraint_3(x1): return 6 - x1  # x₁ + x₂ ≤ 6 → x₂ ≤ 6-x₁


def constraint_4(x1): return (3 * x1 - 12) / 4  # 3x₁ - 4x₂ ≤ 12 → x₂ ≥ (3x₁-12)/4


def constraint_5(x1): return (12 + 4 * x1) / 3  # -4x₁ + 3x₂ ≤ 12 → x₂ ≤ (12+4x₁)/3


y1 = constraint_1(x1_range)
y2 = constraint_2(x1_range)
y3 = constraint_3(x1_range)
y4 = constraint_4(x1_range)
y5 = constraint_5(x1_range)

plt.plot(x1_range, y1, 'r-', linewidth=2, label='x₁ + 5x₂ ≥ 5')
plt.plot(x1_range, y2, 'g-', linewidth=2, label='5x₁ + x₂ ≥ 5')
plt.plot(x1_range, y3, 'b-', linewidth=2, label='x₁ + x₂ ≤ 6')
plt.plot(x1_range, y4, 'm-', linewidth=2, label='3x₁ - 4x₂ ≤ 12')
plt.plot(x1_range, y5, 'c-', linewidth=2, label='-4x₁ + 3x₂ ≤ 12')

plt.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.7, label='x₂ ≥ 0')
plt.axvline(x=0, color='black', linewidth=1, linestyle='-', alpha=0.7, label='x₁ ≥ 0')

print("\nЗНАХОДЖЕННЯ ВЕРШИН ОДР:")

vertices = []

intersections = [
    # Коефіцієнти рівнянь у вигляді [a, b] для ax₁ + bx₂ = c
    ([1, 5], [5, 1], [5, 5]),  # x₁ + 5x₂ = 5 і 5x₁ + x₂ = 5
    ([1, 5], [1, 1], [5, 6]),  # x₁ + 5x₂ = 5 і x₁ + x₂ = 6
    ([1, 5], [3, -4], [5, 12]),  # x₁ + 5x₂ = 5 і 3x₁ - 4x₂ = 12
    ([1, 5], [-4, 3], [5, 12]),  # x₁ + 5x₂ = 5 і -4x₁ + 3x₂ = 12
    ([5, 1], [1, 1], [5, 6]),  # 5x₁ + x₂ = 5 і x₁ + x₂ = 6
    ([5, 1], [3, -4], [5, 12]),  # 5x₁ + x₂ = 5 і 3x₁ - 4x₂ = 12
    ([5, 1], [-4, 3], [5, 12]),  # 5x₁ + x₂ = 5 і -4x₁ + 3x₂ = 12
    ([1, 1], [3, -4], [6, 12]),  # x₁ + x₂ = 6 і 3x₁ - 4x₂ = 12
    ([1, 1], [-4, 3], [6, 12]),  # x₁ + x₂ = 6 і -4x₁ + 3x₂ = 12
    ([3, -4], [-4, 3], [12, 12]),  # 3x₁ - 4x₂ = 12 і -4x₁ + 3x₂ = 12
    # Перетини з осями
    ([1, 0], [1, 5], [0, 5]),  # x₁ = 0 і x₁ + 5x₂ = 5
    ([1, 0], [5, 1], [0, 5]),  # x₁ = 0 і 5x₁ + x₂ = 5
    ([1, 0], [1, 1], [0, 6]),  # x₁ = 0 і x₁ + x₂ = 6
    ([1, 0], [-4, 3], [0, 12]),  # x₁ = 0 і -4x₁ + 3x₂ = 12
    ([0, 1], [1, 5], [5, 0]),  # x₂ = 0 і x₁ + 5x₂ = 5
    ([0, 1], [5, 1], [5, 0]),  # x₂ = 0 і 5x₁ + x₂ = 5
    ([0, 1], [1, 1], [6, 0]),  # x₂ = 0 і x₁ + x₂ = 6
    ([0, 1], [3, -4], [12, 0]),  # x₂ = 0 і 3x₁ - 4x₂ = 12
]


def check_constraints(x1, x2):
    """Перевіряє, чи точка задовольняє всім обмеженням"""
    eps = 1e-10
    return (x1 + 5 * x2 >= 5 - eps and
            5 * x1 + x2 >= 5 - eps and
            x1 + x2 <= 6 + eps and
            3 * x1 - 4 * x2 <= 12 + eps and
            -4 * x1 + 3 * x2 <= 12 + eps and
            x1 >= -eps and x2 >= -eps)


for intersection in intersections:
    A = np.array([intersection[0], intersection[1]])
    b = np.array([intersection[2][0], intersection[2][1]])

    try:
        solution = np.linalg.solve(A, b)
        x1_val, x2_val = solution

        if check_constraints(x1_val, x2_val):
            vertices.append((round(x1_val, 4), round(x2_val, 4)))
    except:
        continue

if check_constraints(0, 0):
    vertices.append((0, 0))

vertices = list(set(vertices))
vertices = sorted(vertices)

print("Знайдені вершини ОДР:")
for i, v in enumerate(vertices):
    print(f"Вершина {i + 1}: x₁ = {v[0]:.4f}, x₂ = {v[1]:.4f}")
    x1, x2 = v
    print(f"  Перевірка: x₁+5x₂={x1 + 5 * x2:.2f}≥5✓, 5x₁+x₂={5 * x1 + x2:.2f}≥5✓, x₁+x₂={x1 + x2:.2f}≤6✓")

if len(vertices) >= 3:
    vertices_array = np.array(vertices)

    center = np.mean(vertices_array, axis=0)
    angles = np.arctan2(vertices_array[:, 1] - center[1], vertices_array[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices_array[sorted_indices]

    # Заливка ОДР
    polygon = Polygon(sorted_vertices, alpha=0.3, color='lightgreen', label='Область допустимих рішень')
    plt.gca().add_patch(polygon)

# Позначаємо вершини на графіку
for i, v in enumerate(vertices):
    plt.plot(v[0], v[1], 'ko', markersize=10)
    plt.annotate(f'V{i + 1}({v[0]:.2f}, {v[1]:.2f})', (v[0], v[1]),
                 xytext=(8, 8), textcoords='offset points', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))



z_values = []
for i, v in enumerate(vertices):
    z = 4 * v[0] - v[1]
    z_values.append(z)
    print(f"Вершина {i + 1}: Z = 4*{v[0]:.4f} - {v[1]:.4f} = {z:.4f}")

if z_values:
    z_min = min(z_values)
    z_max = max(z_values)

    z_levels = np.linspace(z_min - 3, z_max + 3, 6)

    for z_level in z_levels:
        # Z = 4x₁ - x₂ = z_level → x₂ = 4x₁ - z_level
        x1_line = np.linspace(-1, 8, 100)
        x2_line = 4 * x1_line - z_level

        plt.plot(x1_line, x2_line, '--', color='gray', alpha=0.6, linewidth=1)

        # Підпис для лінії рівня
        if len(x1_line) > 50:
            mid_idx = len(x1_line) // 2
            plt.text(x1_line[mid_idx], x2_line[mid_idx], f'Z={z_level:.1f}',
                     rotation=75, fontsize=9, alpha=0.8,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))


optimal_vertex = None
optimal_value = float('-inf')

for i, v in enumerate(vertices):
    z = 4 * v[0] - v[1]
    if z > optimal_value:
        optimal_value = z
        optimal_vertex = v

if optimal_vertex:
    print(f"\nОПТИМАЛЬНЕ РІШЕННЯ:")
    print(f"x₁* = {optimal_vertex[0]:.4f}")
    print(f"x₂* = {optimal_vertex[1]:.4f}")
    print(f"Z* = {optimal_value:.4f} (максимальне значення)")

    # Виділяємо оптимальну точку
    plt.plot(optimal_vertex[0], optimal_vertex[1], 'r*', markersize=20,
             label=f'Оптимум: ({optimal_vertex[0]:.3f}, {optimal_vertex[1]:.3f})')

    # Лінія рівня через оптимальну точку
    z_optimal = optimal_value
    x1_optimal_line = np.linspace(-1, 8, 100)
    x2_optimal_line = 4 * x1_optimal_line - z_optimal

    plt.plot(x1_optimal_line, x2_optimal_line, 'r-', linewidth=3, alpha=0.8,
             label=f'Оптимальна лінія рівня Z={z_optimal:.3f}')

# Налаштування графіка
plt.xlim(-1, 8)
plt.ylim(-1, 8)
plt.xlabel('x₁', fontsize=14)
plt.ylabel('x₂', fontsize=14)
plt.title('Геометричний розв\'язок задачі 14.2\nZ = 4x₁ - x₂ → max', fontsize=16, pad=20)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)

# Виправляємо попередження про layout
plt.subplots_adjust(right=0.95, top=0.9, bottom=0.1, left=0.1)
plt.show()

