from pathlib import Path
from keras.callbacks import BackupAndRestore, ModelCheckpoint, CSVLogger
from data_loading import data_loading
from base_model import base_model
from plots import save_plots

print("Inicjalizacja zmiennych")

base_dir = Path(__file__).resolve().parent
directory = base_dir / "imgs" / "train"

# Ustawienia hiperparametrów modelu
filters_number = 32 # Liczba filtrów w warstwach konwolucyjnych (na początku), co pozwala modelowi uczyć się różnych cech z obrazów
image_size = (224, 224) # Najpopularniejszy rozmiar wejściowy dla modeli CNN (np. ResNet, VGG, Inception)
batch_size = 64 # Optymalny rozmiar wsadu dla GPU i stabilnego oszacowania gradientu, jednocześnie mieszczący się w pamięci RAM/VRAM przy obrazach 224x224
kernel_size = (3, 3) # Najczęściej stosowany rozmiar jądra konwolucyjnego w CNN

train_ds, val_ds = data_loading(directory, image_size, batch_size)  # Wczytanie zbiorów treningowego i walidacyjnego

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

base_model = base_model(input_shape, filters_number, kernel_size)

# Definicja wywołań zwrotnych (callbacks) do monitorowania procesu trenowania modelu
callbacks = [
    BackupAndRestore(backup_dir=base_dir / "backup_models"), # Automatyczne tworzenie kopii zapasowej stanu trenowania modelu, aby można było wznowić trenowanie po przerwaniu
    ModelCheckpoint(filepath=base_dir / "models" / "base_model_checkpoint_v1.keras", save_best_only=True, monitor='val_loss', verbose=1), # Zapisywanie najlepszego modelu na podstawie minimalnej straty walidacyjnej
    CSVLogger(filename=base_dir / "logs" / "base_model_training_log_v1.csv", append=True) # Zapisywanie historii trenowania modelu do pliku CSV
]

base_model_trained = base_model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks) # Trenowanie modelu na zbiorze treningowym i walidacyjnym przez 20 epok

base_model.save(base_dir / "models" / "base_model_trained_v1.keras") # Zapisanie wytrenowanego modelu do pliku HDF5

# Tworzenie i zapisywanie wykresów
save_plots(base_model_trained, filename='base_model_v1_results.png')