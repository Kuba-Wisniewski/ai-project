from analysis import *
from visualisation import *


def main() -> None:
    # Uruchomienie analizy i wizualizacji
    degrees = list(range(1, 100))
    results = bias_variance_analysis(degrees)
    plot_bias_variance_results(results)

    # Przykładowe dopasowania modelu dla stopni 1, 5, 15
    for deg in [1, 5, 15]:
        plot_model_fits(deg)


if __name__ == '__main__':
    main()
