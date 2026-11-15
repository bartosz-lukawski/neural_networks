from keras.models import Sequential
from keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense


def base_model(input_shape=(224, 224, 3), filters_number=32, kernel_size=(3, 3)):
    model = Sequential([
        Input(shape=input_shape),  # Warstwa wejściowa przyjmująca obrazy o rozmiarze 224x224 i 3 kanałach (RGB)
        Rescaling(1./255),  # Normalizacja danych wejściowych do zakresu [0, 1]
    ])

    for i in range(4):
        model.add(Conv2D(filters_number*(2**i), kernel_size, activation='relu')) # Warstwa konwulucyjna z filtrami i funkcją aktywacji ReLU
        model.add(MaxPooling2D((2, 2))) # Zmniejszenie dwukrotnie długości i szerokości mapy cech (obrazu)

    model.add(GlobalAveragePooling2D()) # Zmniejszenie wymiarów mapy cech do jednowymiarowego wektora poprzez uśrednianie, aby ograniczyć liczbę parametrów i uniknąć przeuczenia
    model.add(Dense(10, activation='softmax')) # Warstwa wyjściowa z 10 neuronami (dla 10 klas) i funkcją aktywacji softmax do klasyfikacji wieloklasowej

    model.summary() # Wyświetlenie podsumowania architektury modelu

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Kompilacja modelu z optymalizatorem Adam, funkcją straty dla klasyfikacji wieloklasowej i metryką dokładności

    return model
