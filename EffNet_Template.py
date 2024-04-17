# Importieren notwendiger Bibliotheken
import os
import pandas as pd
import tensorflow as tf
# from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.mixed_precision import set_global_policy
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder


# Aktivieren von Mixed Precision
# Dies ermöglicht es dem Modell, schneller zu trainieren und weniger Speicher zu verbrauchen, 
# indem Berechnungen in geringerer Präzision durchgeführt werden, ohne signifikant die Modellleistung zu beeinträchtigen.
set_global_policy("mixed_float16")

# Modellname für die Benennung der gespeicherten Dateien
modelname = "Testmodell"
 
# Pfad zum aktuellen Verzeichnis und zu den Bildverzeichnissen
base_dir = os.getcwd() # Ermitteln des aktuellen Arbeitsverzeichnispfads

# Die HAM10000-Datenbank ist aufgrund ihrer Größe in zwei Teile aufgeteilt.
# Diese Pfade werden genutzt, um auf die Bilder in beiden Teilen zuzugreifen.
images_dir_part1 = os.path.join(base_dir, "HAM10000_images_part_1")
images_dir_part2 = os.path.join(base_dir, "HAM10000_images_part_2")

 
# Metadaten aus .csv einlesen und Bildpfade hinzufügen (Label zu Bildern)
# Zuerst werden die Metadaten aus der CSV-Datei geladen. Diese Metadaten enthalten wichtige Informationen wie Bild-IDs und entsprechende Labels.
metadata = pd.read_csv(os.path.join(base_dir, "HAM10000_metadata.csv"))

def get_image_path(image_id):
    # Diese Funktion durchsucht die Verzeichnisse, um den vollständigen Pfad für jedes Bild basierend auf seiner ID zu finden.
    # Es ermöglicht eine direkte Verknüpfung zwischen den Metadaten (z.B. Labels) und den tatsächlichen Bilddateien.
    for directory in [images_dir_part1, images_dir_part2]:
        path = os.path.join(directory, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None  # Gibt None zurück, falls das Bild in keinem der Verzeichnisse gefunden wurde.

# Die Bildpfade werden zu den Metadaten hinzugefügt, indem für jede Bild-ID der entsprechende Pfad gesucht wird.
# Dies vereinfacht den Zugriff auf die Bilder für spätere Verarbeitungsschritte, da der Pfad direkt in den Metadaten gespeichert ist.
metadata["path"] = metadata["image_id"].apply(get_image_path)


 
# Label-Encoding für die Klassifikationslabels
# Der LabelEncoder konvertiert die textuellen Labels (Läsionstypen) in numerische Werte.
# Dies ist ein notwendiger Schritt, da maschinelle Lernmodelle mit numerischen Werten und nicht mit Textdaten arbeiten.
# "dx" ist die Spalte in den Metadaten, die die Diagnose (Label für die Hautläsion) enthält.
label_encoder = LabelEncoder()

# Der fit_transform-Prozess lernt die Zuordnung der einzigartigen Label zu ganzen Zahlen und wendet sie gleichzeitig an,
# um eine neue Spalte "label_index" zu erstellen, die die numerischen Labels enthält.
metadata["label_index"] = label_encoder.fit_transform(metadata["dx"])

# Die umgewandelten Labels werden für das Training des Modells verwendet.
labels = metadata["label_index"]

 
# Daten-Augmentation durch Oversampling zur Adressierung von Klassenungleichgewichten
# Oversampling ist eine Technik zur Erhöhung der Anzahl von Beispielen in unterrepräsentierten Klassen.
# Dies kann helfen, Ungleichgewichte in den Label-Verhältnissen auszugleichen.
# Durch das Hinzufügen künstlich erzeugter oder duplizierter Beispiele wird die Wahrscheinlichkeit von Overfitting reduziert.
# Overfitting tritt auf, wenn ein Modell die Trainingsdaten zu genau lernt und sich dadurch schlechter auf neuen Daten verhält.
# Hier wird RandomOverSampler aus der imblearn-Bibliothek verwendet, um ein gleichmäßigeres Verhältnis zwischen den Klassen zu erreichen.
ros = RandomOverSampler()

# "metadata[["path", "label_index"]]" enthält die Pfade und Label-Indizes für das Oversampling.
# Das Ergebnis sind ein neuer Datensatz "metadata_resampled" und die entsprechenden Labels "labels_resampled",
# die ein ausgeglicheneres Klassenverhältnis aufweisen.
metadata_resampled, labels_resampled = ros.fit_resample(metadata[["path", "label_index"]], labels)

# Die neuen, resampelten Labels werden den Metadaten hinzugefügt, um sie für das Training vorzubereiten.
metadata_resampled["label_index"] = labels_resampled

 
# Vorverarbeitung der Bilder für das Modell
# Diese Funktion liest ein Bild von einem gegebenen Pfad, decodiert es zu einem Tensor, ändert seine Größe auf 224x224 Pixel
# und wendet dann eine Modell-spezifische Vorverarbeitung (hier für ResNet50) an. Dieser Schritt ist notwendig, um sicherzustellen,
# dass alle Bilder die gleiche Größe und Formatierung haben, was für die korrekte Funktionsweise des Modells erforderlich ist.
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [400, 290])
    image = preprocess_input(image)  # ResNet Vorverarbeitung
    return image, label

# Erstellen eines TensorFlow Datasets aus den Pfaden und Labels im DataFrame.
# Jedes Element des Datasets wird durch die "preprocess_image" Funktion vorverarbeitet,
# die das Bild lädt, seine Größe anpasst und es für das Training vorbereitet.
def load_and_preprocess_from_dataframe(df):
    labels = to_categorical(df["label_index"], num_classes=len(label_encoder.classes_))
    paths = df["path"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) # num_parallel_calls=tf.data.AUTOTUNE ermöglicht TensorFlow, 
                                                                       # automatisch die optimale Anzahl von Prozessen für das Laden 
                                                                       # und Vorverarbeiten der Bilder zu bestimmen,
                                                                       # was zu einer schnelleren Datenverarbeitung führt.
    return ds


# Hier wird der resampelte DataFrame "metadata_resampled" in Trainings- und Validierungsdatensätze aufgeteilt.
train_df, validation_df = train_test_split(metadata_resampled, test_size=0.03, random_state=42) # Etwa 500 Bilder in den Validierungsdaten, da insgesamt 10000 Bilder

# Die Aufteilung ermöglicht eine unabhängige Bewertung der Modellleistung auf Daten, die während des Trainings nicht gesehen wurden.


# Trainings- und Validierungsdatensatz Laden und Vorverarbeiten durch vorherig definierte Funktionen
# Nachdem die Trainings- und Validierungsdatenframes erstellt wurden, wird die zuvor definierte Funktion 
# "load_and_preprocess_from_dataframe" verwendet um sie in TensorFlow Datasets umzuwandeln. 
ds_train = load_and_preprocess_from_dataframe(train_df)
ds_validation = load_and_preprocess_from_dataframe(validation_df)

# Jetzt sind die Daten einheitlich und somit geeignet für das Training


# Durchwürfeln eines Blocks von 256 Bildern um Overfitting zu vermeiden
# Das Trainingsdataset wird mit "shuffle" durchgemischt mit einer Puffergröße gleich der Länge des Trainingsdatenframes,
# um sicherzustellen, dass die Daten gut durchmischt werden. Dies hilft, das Modell generischer zu machen und Overfitting zu reduzieren.
# Anschließend wird das Dataset in Batches von 256 Bildern unterteilt ("batch"), was die Größe der Datenblöcke angibt,
# die gleichzeitig verarbeitet werden. Größere Batches können die Trainingsgeschwindigkeit verbessern, benötigen aber mehr Speicher.
# "prefetch" wird verwendet, um die Ladezeit zu verkürzen, indem im Voraus Daten für das Training bereitgestellt werden.
# Dabei ermöglicht "tf.data.AUTOTUNE", dass TensorFlow automatisch die optimale Anzahl von Batches bestimmt, die im Voraus geladen werden sollen.
ds_train = ds_train.shuffle(buffer_size=len(train_df)).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)
ds_validation = ds_validation.batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)


# Definition des Modells, das für die Klassifikation von Hautläsionen eingesetzt wird. 
# Ein vortrainiertes ResNet50-Modell wird als Basis verwendet, um den Rechenaufwand zu verringern und von bereits gelernten Merkmalen zu profitieren.
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(400, 290, 3))  # Vortrainiertes Modell lädt Gewichte von ImageNet "imagenet" anstatt None



# Hinzufügen von Schichten zur Anpassung des Modells an das spezifische Problem:
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduzierung der Bildauflösung zur Verringerung des Rechenaufwands
x = Dense(2084, activation="relu")(x)  # Eine dichte Schicht, die den Output zusammenfasst, um ein klar definiertes Ergebnis zu liefern

# Optional: Ein Dropout kann eingeführt werden, um Overfitting zu verringern. Wurde in meinem Fall durch Data Augmentation obsolet.
#x = Dropout(0.3)(x)

# Die finale Schicht ordnet das Ergebnis den entsprechenden Labelnamen zu. Hier wird "softmax" für die Klassifikation verwendet.
predictions = Dense(len(label_encoder.classes_), activation="softmax", dtype="float32")(x)

# Zusammenstellung des Modells, das die Eingaben des Basis-Modells und die Ausgaben der vorhergesagten Klassifikation nutzt.
model = Model(inputs=base_model.input, outputs=predictions)  # Initialisierung des Modells mit den spezifizierten Ein- und Ausgängen


# Einfrieren des Basis-Modells, um Aktualisierungen der Gewichte während des Trainings zu verhindern.
# Dies ermöglicht die Nutzung vortrainierter Merkmale ohne zusätzlichen Rechenaufwand.
for layer in base_model.layers:
    layer.trainable = True


# Kompilierung des Modells mit definierter Lernrate, Verlustfunktion und Leistungsmetriken.
# Adam-Optimierer mit einer angepassten, niedrigen Lernrate für präzise Anpassungen.
# "categorical_crossentropy" wird für Mehrklassen-Klassifikationsaufgaben verwendet.
# "accuracy" dient als Metrik zur Bewertung der Modellleistung.
model.compile(optimizer=Adam(learning_rate=0.002), loss="categorical_crossentropy", metrics=["accuracy"])
# Die aktuelle Lernrate (learning_rate) wurde durch iteratives Anpassen ermittelt.


# Einrichtung des frühen Anhaltens als Maßnahme gegen Overfitting und zur Vermeidung unproduktiver Trainingsschritte.
# Durch Überwachung des "val_loss" wird das Training gestoppt, wenn sich die Validierungsverluste für eine festgelegte 
# Anzahl von Epochen ("patience" = 10) nicht verbessern. Dies hilft, Ressourcen effizient zu nutzen und das Modell vor Overfitting zu schützen.
#early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)


class CustomLREarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, min_epochs=25, patience=5):
        super(CustomLREarlyStopping, self).__init__()
        self.min_epochs = min_epochs  # Minimale Anzahl von Epochen vor der Überprüfung
        self.patience = patience  # Wie viele Epochen zu warten, nachdem min_epochs erreicht sind, falls sich lr nicht ändert
        self.wait = 0  # Zählt, wie viele Epochen die lr unverändert blieb, nachdem min_epochs erreicht wurden
        self.lr_changes = 0  # Zählt, wie viele Male sich die lr geändert hat

    def on_epoch_end(self, epoch, logs=None):
        current_lr = self.model.optimizer.lr.numpy()  # Aktuelle Lernrate holen
        if epoch == 0:
            self.last_lr = current_lr  # Initiale Lernrate setzen
        
        # Wenn die aktuelle Epoche >= min_epochs ist, beginnen wir zu prüfen, ob sich die lr ändert
        if epoch >= self.min_epochs:
            if current_lr != self.last_lr:
                self.lr_changes += 1  # Zähler für lr-Änderungen erhöhen
                self.wait = 0  # Warte-Zähler zurücksetzen, da sich die lr geändert hat
            else:
                self.wait += 1  # Warte-Zähler erhöhen, wenn lr unverändert bleibt

            # Training beenden, wenn die lr für 'patience' Epochen nach 'min_epochs' unverändert bleibt
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nTraining gestoppt, da sich die Lernrate für {self.patience} Epochen nach den ersten {self.min_epochs} Epochen nicht geändert hat.")
        
        self.last_lr = current_lr  # Aktualisiere die zuletzt bekannte Lernrate für den nächsten Durchgang


lr_early_stopping = CustomLREarlyStopping(min_epochs=25, patience=5)


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=modelname+"_modell_epoch_{epoch:02d}_val_accuracy_{val_accuracy:.4f}.h5",  # Pfad, wo das Modell gespeichert wird
    save_best_only=True,  # Nur das beste Modell wird gespeichert
    monitor="val_accuracy",  # Bestes Modell basierend auf der höchsten val_accuracy
    mode="max",  # "max" bedeutet, dass höhere val_accuracy als besser angesehen wird
    verbose=1  # Zeigt eine Nachricht jedes Mal, wenn das Modell aktualisiert (überschrieben) wird
)


# Callback zum Speichern der Trainingshistorie nach jeder Epoche in einer JSON-Datei
import json
import numpy as np

class HistoryLogger(tf.keras.callbacks.Callback):
    def __init__(self, path="{modelname}training_history.json"):
        super().__init__()
        self.path = path
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        # Konvertiere alle Werte in float, um Kompatibilität mit JSON zu gewährleisten
        logs = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in logs.items()}
        logs['epoch'] = epoch + 1
        self.history.append(logs)
        
        # Stelle sicher, dass das Verzeichnis existiert
        dir_name = os.path.dirname(self.path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        
        # Speichere die komplette Historie nach jeder Epoche
        with open(self.path, 'w') as f:
            json.dump(self.history, f, indent=4)

history_logger_callback = HistoryLogger("{modelname}training_history.json")


# Training des Modells für eine festgelegte Anzahl von Epochen. Hier wird der zuvor definierte EarlyStopping-Callback verwendet.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_lr=0.00000001) # Reduzierung der LR während des Trainings

history = model.fit(
                    ds_train,
                    epochs=3,
                    validation_data=ds_validation,
                    callbacks=[lr_early_stopping, reduce_lr, model_checkpoint_callback, history_logger_callback]
                    )


# Erstellung eines Dictionarys mit zusammengefassten Trainingsinformationen.
# Dies beinhaltet die Anzahl der durchlaufenen Epochen und die finale Trainings- sowie Validierungsleistung.
training_info = {
    "epochs": len(history.history["loss"]),
    "final_train_loss": history.history["loss"][-1],
    "final_train_accuracy": history.history["accuracy"][-1],
    "final_val_loss": history.history["val_loss"][-1],
    "final_val_accuracy": history.history["val_accuracy"][-1],
}


# Formattieren der finalen Validierungsverlust und -genauigkeit auf zwei Nachkommastellen für die Benennung des gespeicherten Modells.
formatted_loss = "{:.4f}".format(training_info["final_val_loss"])
formatted_accuracy = "{:.4f}".format(training_info["final_val_accuracy"])

# Speichern des trainierten Modells unter einem aussagekräftigen Dateinamen, der wichtige Metriken beinhaltet.
model_filename = f"{modelname}_model_val_loss_{formatted_loss}_val_acc_{formatted_accuracy}_epochs_{training_info['epochs']}.h5"
model.save("modelle/" + model_filename)

print(f"Modell gespeichert als: {model_filename}")


# Speichern der Trainingsinformationen in eine JSON-Datei, um eine einfache Nachvollziehbarkeit und Auswertung zu ermöglichen.
import json
json_filename = f"modelle/{modelname}_training_info_{formatted_loss}_acc_{formatted_accuracy}_epochs_{training_info['epochs']}.json"
with open(json_filename, "w") as f:
    json.dump(training_info, f)

print(f"Trainingsinformationen gespeichert als: {json_filename}")


import joblib
# Der LabelEncoder wird im selben Verzeichnis "modelle" gespeichert, um Konsistenz zu gewährleisten.
# Dies ist wichtig für die spätere Wiederverwendung des Modells, da der LabelEncoder benötigt wird, um Vorhersagen zu interpretieren.
joblib.dump(label_encoder, "modelle/{modelname}_label_encoder.pkl")

# Ausgabe zur Bestätigung, dass der LabelEncoder gespeichert wurde.
print("LabelEncoder gespeichert unter: modelle/{modelname}label_encoder.pkl")


from sklearn.metrics import classification_report
import numpy as np

# Vorhersagen für den gesamten Validierungsdatensatz werden gesammelt
predictions = model.predict(ds_validation)
predicted_classes = np.argmax(predictions, axis=1)


# Tatsächliche Labels werden extrahiert
true_labels = np.concatenate([y for x, y in ds_validation], axis=0)
true_classes = np.argmax(true_labels, axis=1)


# Klassennamen aus dem LabelEncoder wird extrahiert
class_names = label_encoder.inverse_transform(range(len(label_encoder.classes_)))


# Bericht generieren um einen Überblick der Testergebmnisse zu erhalten
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print(report)

# Formatierung des Dateinamens für den Bericht
report_filename = f"modelle/{modelname}_classification_report_loss_{formatted_loss}_acc_{formatted_accuracy}_epochs_{training_info['epochs']}.txt"

# Speichern des Berichts in eine Textdatei
with open(report_filename, "w") as text_file:
    text_file.write(report)

print(f"Klassifikationsbericht wurde als '{report_filename}' gespeichert.")
