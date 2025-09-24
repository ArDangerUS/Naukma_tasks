import numpy as np

def thomas_algorithm(a, b, c, d):
    n = len(d)
    alpha = np.zeros(n)
    beta = np.zeros(n)

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * alpha[i - 1]
        if denom == 0:
            raise ValueError("Ділення на нуль у методі прогонки!")
        alpha[i] = -c[i] / denom if i < n - 1 else 0
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x

if __name__ == "__main__":
    # Приклад 1
    a3 = np.array([0, -1, -1, -1], dtype=float)  # піддіагональ
    b3 = np.array([2, 2, 2, 2], dtype=float)     # головна діагональ
    c3 = np.array([-1, -1, -1, 0], dtype=float)  # наддіагональ
    d3 = np.array([1, 0, 0, 6], dtype=float)     # права частина

    x3 = thomas_algorithm(a3, b3, c3, d3)
    print("Розв'язок прикладу 1:", x3)
