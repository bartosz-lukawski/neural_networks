import keras
import tensorflow as tf
from pathlib import Path
from keras.models import Sequential
from keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense

print("Inicjalizacja zmiennych")

base_dir = Path(__file__).resolve().parent
directory = base_dir / "imgs" / "train"

# Ustawienia hiperparametrów modelu
filters_number = 32 # Liczba filtrów w warstwach konwolucyjnych (na początku), co pozwala modelowi uczyć się różnych cech z obrazów
image_size = (224, 224) # Najpopularniejszy rozmiar wejściowy dla modeli CNN (np. ResNet, VGG, Inception)
batch_size = 64 # Optymalny rozmiar wsadu dla GPU i stabilnego oszacowania gradientu, jednocześnie mieszczący się w pamięci RAM/VRAM przy obrazach 224x224
kernel_size = (3, 3) # Najczęściej stosowany rozmiar jądra konwolucyjnego w CNN

# print(base_dir)
# print(directory)
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

# Sprawdzenie wykrytych klas
print("Wykryte klasy:", train_ds.class_names)

# Do odpalenie na GPU, jeśli jest dostępne (równoległe ładowanie danych)
# train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE) # Prefetching danych w celu zwiększenia wydajności trenowania modelu poprzez równoległe ładowanie danych podczas trenowania. AUTOTUNE pozwala TensorFlow automatycznie dostosować rozmiar bufora do dostępnych zasobów systemowych.

# Do odpalenie, jeśli jest dostępne GPU i wystarczająca ilość pamięci RA
# train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Na sprzęcie bez GPU i małą ilością RAM zostawić originalne train_ds

input_shape = (image_size[0], image_size[1], 3) # Kształt danych wejściowych (wysokość, szerokość, liczba kanałów)

# input_layer = keras.Input(shape=input_shape) # Definicja warstwy wejściowej modelu

# rescaling = keras.layers.Rescaling(1./255) # Normalizacja danych wejściowych do zakresu [0, 1]

# normalized_image = rescaling(input_shape)

# sequential_model = keras.Sequential()
# sequential_model.add(normalized_image)

base_model = Sequential([
    Input(shape=input_shape),  # Warstwa wejściowa przyjmująca obrazy o rozmiarze 224x224 i 3 kanałach (RGB)
    Rescaling(1./255),  # Normalizacja danych wejściowych do zakresu [0, 1]
])

for i in range(4):
    base_model.add(Conv2D(filters_number*(2**i), kernel_size, activation='relu')) # Warstwa konwulucyjna z filtrami i funkcją aktywacji ReLU
    base_model.add(MaxPooling2D((2, 2))) # Zmniejszenie dwukrotnie długości i szerokości mapy cech (obrazu)

base_model.add(GlobalAveragePooling2D()) # Zmniejszenie wymiarów mapy cech do jednowymiarowego wektora poprzez uśrednianie, aby ograniczyć liczbę parametrów i uniknąć przeuczenia
base_model.add(Dense(10, activation='softmax')) # Warstwa wyjściowa z 10 neuronami (dla 10 klas) i funkcją aktywacji softmax do klasyfikacji wieloklasowej

base_model.summary() # Wyświetlenie podsumowania architektury modelu

base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Kompilacja modelu z optymalizatorem Adam, funkcją straty dla klasyfikacji wieloklasowej i metryką dokładności

history = base_model.fit(train_ds, validation_data=val_ds, epochs=20) # Trenowanie modelu na zbiorze treningowym i walidacyjnym przez 20 epok

# Przykład z gemini:
# model = Sequential([
#     Input(shape=(10,)),            # Warstwa wejściowa przyjmującawektory o długości 10
#     Dense(32, activation='relu'),  # Pierwsza warstwa ukryta z 32 neuronami i funkcją ReLU
#     Dense(16, activation='relu'),  # Druga warstwa ukryta z 16 neuronami
#     Dense(1, activation='sigmoid') # Warstwa wyjściowa z 1 neuronem (idealna do klasyfikacji binarnej)
# ])