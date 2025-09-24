import math


def newton_method(f, df, x0, eps=1e-6, max_iterations=100):
    x = x0

    for k in range(max_iterations):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            print("Derivative too small, method may fail")
            break

        x_new = x - fx / dfx

        print(f"Iteration {k + 1}: x = {x_new:.8f}, f(x) = {f(x_new):.2e}")

        if abs(x_new - x) < eps:
            print(f"Converged after {k + 1} iterations")
            return x_new

        x = x_new

    print("Maximum iterations reached")
    return x


# Example usage
if __name__ == "__main__":
    # Example 1: x^2 - 2 = 0 (root is sqrt(2))
    def f1(x):
        return x ** 2 - 2


    def df1(x):
        return 2 * x


    print("Example 1: x^2 - 2 = 0")
    root1 = newton_method(f1, df1, x0=1.0, eps=1e-8)
    print(f"Root: {root1:.8f}")
    print(f"Verification: f({root1:.8f}) = {f1(root1):.2e}\n")


    # Example 2: cos(x) - x = 0
    def f2(x):
        return math.cos(x) - x


    def df2(x):
        return -math.sin(x) - 1


    print("Example 2: cos(x) - x = 0")
    root2 = newton_method(f2, df2, x0=0.5, eps=1e-8)
    print(f"Root: {root2:.8f}")
    print(f"Verification: f({root2:.8f}) = {f2(root2):.2e}\n")


    # Example 3: x^3 - 2x - 5 = 0
    def f3(x):
        return x ** 3 - 2 * x - 5


    def df3(x):
        return 3 * x ** 2 - 2


    print("Example 3: x^3 - 2x - 5 = 0")
    root3 = newton_method(f3, df3, x0=2.0, eps=1e-8)
    print(f"Root: {root3:.8f}")
    print(f"Verification: f({root3:.8f}) = {f3(root3):.2e}\n")


    # Example 4: x^3 - 6x^2 + 11x - 6 = 0 (has roots at x=1,2,3)
    def f4(x):
        return x ** 3 - 6 * x ** 2 + 11 * x - 6


    def df4(x):
        return 3 * x ** 2 - 12 * x + 11


    print("Example 4: x^3 - 6x^2 + 11x - 6 = 0 (finding root near x=1)")
    root4a = newton_method(f4, df4, x0=0.5, eps=1e-8)
    print(f"Root: {root4a:.8f}")
    print(f"Verification: f({root4a:.8f}) = {f4(root4a):.2e}\n")

    print("Example 4b: Same equation, finding root near x=2")
    root4b = newton_method(f4, df4, x0=1.8, eps=1e-8)
    print(f"Root: {root4b:.8f}")
    print(f"Verification: f({root4b:.8f}) = {f4(root4b):.2e}\n")

    print("Example 4c: Same equation, finding root near x=3")
    root4c = newton_method(f4, df4, x0=3.2, eps=1e-8)
    print(f"Root: {root4c:.8f}")
    print(f"Verification: f({root4c:.8f}) = {f4(root4c):.2e}\n")


    # Example 5: x^3 + x - 1 = 0 (one real root)
    def f5(x):
        return x ** 3 + x - 1


    def df5(x):
        return 3 * x ** 2 + 1


    print("Example 5: x^3 + x - 1 = 0")
    root5 = newton_method(f5, df5, x0=0.5, eps=1e-8)
    print(f"Root: {root5:.8f}")
    print(f"Verification: f({root5:.8f}) = {f5(root5):.2e}\n")


    # Example 6: x^3 - 3x + 1 = 0 (three real roots)
    def f6(x):
        return x ** 3 - 3 * x + 1


    def df6(x):
        return 3 * x ** 2 - 3


    print("Example 6: x^3 - 3x + 1 = 0 (finding different roots)")

    print("Root near x=-2:")
    root6a = newton_method(f6, df6, x0=-2.0, eps=1e-8)
    print(f"Root: {root6a:.8f}")
    print(f"Verification: f({root6a:.8f}) = {f6(root6a):.2e}\n")

    print("Root near x=0:")
    root6b = newton_method(f6, df6, x0=0.3, eps=1e-8)
    print(f"Root: {root6b:.8f}")
    print(f"Verification: f({root6b:.8f}) = {f6(root6b):.2e}\n")

    print("Root near x=2:")
    root6c = newton_method(f6, df6, x0=2.0, eps=1e-8)
    print(f"Root: {root6c:.8f}")
    print(f"Verification: f({root6c:.8f}) = {f6(root6c):.2e}\n")


# Interactive version
def solve_equation():
    print("\nEnter your equation as Python expression (use 'x' as variable)")
    print("Example: x**3 - 2*x - 5 for x^3 - 2x - 5 = 0")
    print("Example: x**2 - 2 for x^2 - 2 = 0")

    f_expr = input("f(x) = ")
    df_expr = input("f'(x) = ")

    def f(x):
        return eval(f_expr)

    def df(x):
        return eval(df_expr)

    x0 = float(input("Initial guess x0: "))
    eps = float(input("eps (e.g., 1e-6): "))

    root = newton_method(f, df, x0, eps)
    print(f"\nFinal result: x = {root:.8f}")

# Uncomment to run interactive mode
# solve_equation()