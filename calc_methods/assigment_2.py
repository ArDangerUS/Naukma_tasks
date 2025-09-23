import math


def newton_method(f, df, x0, tolerance=1e-6, max_iterations=100):
    """
    Newton's method for solving f(x) = 0
    f: function
    df: derivative of f
    x0: initial guess
    tolerance: convergence tolerance
    max_iterations: maximum number of iterations
    """
    x = x0

    for k in range(max_iterations):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            print("Derivative too small, method may fail")
            break

        x_new = x - fx / dfx

        print(f"Iteration {k + 1}: x = {x_new:.8f}, f(x) = {f(x_new):.2e}")

        if abs(x_new - x) < tolerance:
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
    root1 = newton_method(f1, df1, x0=1.0, tolerance=1e-8)
    print(f"Root: {root1:.8f}")
    print(f"Verification: f({root1:.8f}) = {f1(root1):.2e}\n")


    # Example 2: cos(x) - x = 0
    def f2(x):
        return math.cos(x) - x


    def df2(x):
        return -math.sin(x) - 1


    print("Example 2: cos(x) - x = 0")
    root2 = newton_method(f2, df2, x0=0.5, tolerance=1e-8)
    print(f"Root: {root2:.8f}")
    print(f"Verification: f({root2:.8f}) = {f2(root2):.2e}")


# Interactive version
def solve_equation():
    print("\nEnter your equation as Python expression (use 'x' as variable)")
    print("Example: x**2 - 2 for x^2 - 2 = 0")

    f_expr = input("f(x) = ")
    df_expr = input("f'(x) = ")

    def f(x):
        return eval(f_expr)

    def df(x):
        return eval(df_expr)

    x0 = float(input("Initial guess x0: "))
    tolerance = float(input("Tolerance (e.g., 1e-6): "))

    root = newton_method(f, df, x0, tolerance)
    print(f"\nFinal result: x = {root:.8f}")

# solve_equation()