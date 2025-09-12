import math


# Метод ділення навпіл
def bisection(f, a, b, eps=1e-6):
    if f(a) * f(b) > 0:
        raise ValueError("На відрізку немає кореня або їх парна кількість")

    iterations = 0
    while (b - a) / 2 > eps:
        iterations += 1
        mid = (a + b) / 2
        if f(mid) == 0:
            return mid, iterations
        elif f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2, iterations


# Метод простої ітерації
def simple_iteration(phi, x0, eps=1e-6, max_iter=1000):
    x = x0
    for i in range(1, max_iter + 1):
        x_next = phi(x)
        if abs(x_next - x) < eps:
            return x_next, i
        x = x_next
    raise ValueError("Метод не збігся за вказану кількість ітерацій")


if __name__ == "__main__":
    f1 = lambda x: x ** 3 - x - 2
    root_bisect1, it1 = bisection(f1, 1, 2, eps=1e-6)
    print("Корінь (бісекція, приклад 1):", root_bisect1, "ітерацій:", it1)

    phi1 = lambda x: (x + 2) ** (1 / 3)
    root_iter1, it2 = simple_iteration(phi1, x0=1.5, eps=1e-6)
    print("Корінь (ітерації, приклад 2):", root_iter1, "ітерацій:", it2)

    f2 = lambda x: math.cos(x) - x
    root_bisect2, it3 = bisection(f2, 0, 1, eps=1e-6)
    print("Корінь (бісекція, приклад 3):", root_bisect2, "ітерацій:", it3)

    phi2 = lambda x: math.cos(x)
    root_iter2, it4 = simple_iteration(phi2, x0=0.5, eps=1e-6)
    print("Корінь (ітерації, приклад 4):", root_iter2, "ітерацій:", it4)

    f3 = lambda x: x ** 2 - 2
    root_bisect3, it5 = bisection(f3, 1, 2, eps=1e-6)
    print("Корінь (бісекція, приклад 5):", root_bisect3, "ітерацій:", it5)

    phi3 = lambda x: (2) ** 0.5
    root_iter3, it6 = simple_iteration(phi3, x0=1.0, eps=1e-6)
    print("Корінь (ітерації, приклад 6):", root_iter3, "ітерацій:", it6)
