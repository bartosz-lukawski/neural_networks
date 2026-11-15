
import keras

def data_loading(directory, image_size, batch_size):
    print("Wczytywanie zbioru treningowego")
    # Wczytanie zbioru treningowego (80% danych)
    train_ds = keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2, # Użycie 20% danych jako zbioru walidacyjnego
        subset="training",
        seed=100,  # Stałe ziarno dla zapewnienia braku wycieku danych między zbiorami
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical' # categorical zamiast int, aby łatwiej później było stosować np. smoothing
    )

    print("Wczytywanie zbioru walidacyjnego")
    # Wczytanie zbioru walidacyjnego (20% danych)
    val_ds = keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="validation",
        seed=100, # Stałe ziarno dla zapewnienia braku wycieku danych między zbiorami
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical' # categorical zamiast int, aby łatwiej później było stosować np. smoothing
    )

    return train_ds, val_ds