

# ===============================
# STEP 1: Imports
# ===============================
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf

# ===============================
# STEP 2: Dataset Path
# ===============================
DATASET_PATH = "data"  # Replace with your dataset folder path

# ===============================
# STEP 3: Feature Extraction (MFCC)
# ===============================
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # Pad / truncate
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print("❌ Error processing file:", file_path, e)
        return None

# ===============================
# STEP 4: Load Dataset
# ===============================
labels = []
features = []

for label in ["cry", "not_cry"]:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        mfccs = extract_features(file_path)
        if mfccs is not None:
            features.append(mfccs)
            labels.append(label)

X = np.array(features)
y = np.array(labels)

print("✅ Feature shape:", X.shape)
print("✅ Labels shape:", y.shape)

# ===============================
# STEP 5: Encode Labels
# ===============================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

# Add channel dimension for CNN
X = X[..., np.newaxis]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

print("✅ Train shape:", X_train.shape, y_train.shape)
print("✅ Test shape:", X_test.shape, y_test.shape)

# ===============================
# STEP 6: Build CNN Model
# ===============================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(40, 174, 1)),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===============================
# STEP 7: Train Model
# ===============================
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ===============================
# STEP 8: Evaluate Model
# ===============================
test_loss, test_acc = model.evaluate(X_test, y_test)
print("🎯 Test Accuracy:", test_acc)

# ===============================
# STEP 9: Save model for API usage
# ===============================
MODEL_PATH = "cry_detector.keras"  # or .h5
model.save(MODEL_PATH)
print(f"💾 Model saved for API use: {MODEL_PATH}")
