import matplotlib.pyplot as plt


def plot_loss(ax, epoki, dane):
    # Wykres strat treningowych
    ax.plot(epoki, dane['loss'], label='Strata treningowa')  # Wykres strat treningowych
    ax.plot(epoki, dane['val_loss'], 'r--o', label='Walidacja (val_loss)')
    ax.set_title('Funkcja straty w zależności od epoki')
    ax.set_xlabel('Epoka')
    ax.set_ylabel('Strata')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_accuracy(ax, epoki, dane):
    # Wykres dokładności treningowej
    ax.plot(epoki, dane['accuracy'], label='Dokładność treningowa')
    ax.plot(epoki, dane['val_accuracy'], 'r--o', label='Walidacja (val_accuracy)')
    ax.set_title('Dokładność w zależności od epoki')
    ax.set_xlabel('Epoka')
    ax.set_ylabel('Dokładność')
    ax.legend()
    ax.grid(True, alpha=0.3)


def save_plots(history, filename='base_model_v1_results.png'):
    dane = history.history  # Zapisanie historii trenowania modelu, w tym strat i dokładności dla każdej epoki
    epoki = range(1, len(dane['loss']) + 1)  # Zakres epok od 1 do liczby epok

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # Utworzenie wykresów dla strat i dokładności

    # Wykres strat treningowych
    plot_loss(ax1, epoki, dane)

    # Wykres dokładności treningowej
    plot_accuracy(ax2, epoki, dane)

    plt.tight_layout()
    plt.savefig(filename)  # Zapisanie wykresów do pliku PNG
