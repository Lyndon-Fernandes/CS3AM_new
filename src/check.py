# def cubic_function(x):
#     # Coefficients of the cubic polynomial
#     a = 2.2944444444444407
#     b = -26.595833333333275
#     c = 123.53710317460295
#     d = -183.80928571428558

#     # Calculate y based on the polynomial equation
#     y = a * (x**3) + b * (x**2) + c * x + d
#     return y


# print(cubic_function(10))

from scipy.optimize import fsolve


def inverse_cubic_function(y):
    # Coefficients of the cubic polynomial
    a = 2.2944444444444407
    b = -26.595833333333275
    c = 123.53710317460295
    d = -183.80928571428558

    # Define the equation to solve: a*x^3 + b*x^2 + c*x + d - y = 0
    equation = lambda x: a * (x**3) + b * (x**2) + c * x + d - y

    # Use fsolve to find the root of the equation
    x_initial_guess = 5  # A reasonable initial guess
    x_solution = fsolve(equation, x_initial_guess)

    return x_solution[0]  # Return the first (and usually only) solution


# Example usage
iterations = 8 * 60 * 60
print(round(inverse_cubic_function(iterations), 2))
