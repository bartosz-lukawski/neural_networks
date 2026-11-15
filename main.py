import keras
import matplotlib
import tensorflow as tf
from pathlib import Path
from keras.models import Sequential
from keras.layers import Input, Rescaling, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
import matplotlib.pyplot as plt
from keras.callbacks import BackupAndRestore, ModelCheckpoint, CSVLogger

print("Inicjalizacja zmiennych")

base_dir = Path(__file__).resolve().parent
directory = base_dir / "imgs" / "train"

# Ustawienia hiperparametrów modelu
filters_number = 32 # Liczba filtrów w warstwach konwolucyjnych (na początku), co pozwala modelowi uczyć się różnych cech z obrazów
image_size = (224, 224) # Najpopularniejszy rozmiar wejściowy dla modeli CNN (np. ResNet, VGG, Inception)
batch_size = 64 # Optymalny rozmiar wsadu dla GPU i stabilnego oszacowania gradientu, jednocześnie mieszczący się w pamięci RAM/VRAM przy obrazach 224x224
kernel_size = (3, 3) # Najczęściej stosowany rozmiar jądra konwolucyjnego w CNN

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

# Do odpalenie, jeśli jest dostępne GPU i wystarczająca ilość pamięci RAM
# train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Do odpalenia na sprzęcie bez GPU i wykorzystując cache procesora
# train_ds = train_ds.cache()

# Na sprzęcie bez GPU i małą ilością RAM zostawić originalne train_ds

input_shape = (image_size[0], image_size[1], 3) # Kształt danych wejściowych (wysokość, szerokość, liczba kanałów)

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

# Definicja wywołań zwrotnych (callbacks) do monitorowania procesu trenowania modelu
callbacks = [
    BackupAndRestore(backup_dir=base_dir / "backup_models"), # Automatyczne tworzenie kopii zapasowej stanu trenowania modelu, aby można było wznowić trenowanie po przerwaniu
    ModelCheckpoint(filepath=base_dir / "models" / "base_model_checkpoint_v1.keras", save_best_only=True, monitor='val_loss', verbose=1), # Zapisywanie najlepszego modelu na podstawie minimalnej straty walidacyjnej
    CSVLogger(filename=base_dir / "logs" / "base_model_training_log_v1.csv", append=True) # Zapisywanie historii trenowania modelu do pliku CSV
]

base_model_trained = base_model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks) # Trenowanie modelu na zbiorze treningowym i walidacyjnym przez 20 epok

base_model.save(base_dir / "models" / "base_model_trained_v1.keras") # Zapisanie wytrenowanego modelu do pliku HDF5

dane = base_model_trained.history  # Zapisanie historii trenowania modelu, w tym strat i dokładności dla każdej epoki
epoki = range(1, len(dane['loss']) + 1)  # Zakres epok od 1 do liczby epok

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # Utworzenie wykresów dla strat i dokładności

# Wykres strat treningowych
ax1.plot(epoki, dane['loss'], label='Strata treningowa')  # Wykres strat treningowych
ax1.plot(epoki, dane['val_loss'], 'r--o', label='Walidacja (val_loss)')
ax1.set_title('Funkcja straty w zależności od epoki')
ax1.set_xlabel('Epoka')
ax1.set_ylabel('Strata')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Wykres dokładności treningowej
ax2.plot(epoki, dane['accuracy'], label='Dokładność treningowa')
ax2.plot(epoki, dane['val_accuracy'], 'r--o', label='Walidacja (val_accuracy)')
ax2.set_title('Dokładność w zależności od epoki')
ax2.set_xlabel('Epoka')
ax2.set_ylabel('Dokładność')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()