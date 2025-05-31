from analysis import *
from visualisation import *


def main() -> None:

    # bias-variance
    degrees = np.asarray(range(1, 21))
    results = bias_variance_analysis(degrees)
    plot_bias_variance_results(results)

    # model fits for polynomial degree of 1, 5, 15
    for deg in [1, 5, 15]:
        x_test, y_test, results = fitness_analysis(deg)
        plot_model_fits(deg, x_test, y_test, results)


if __name__ == '__main__':
    main()
