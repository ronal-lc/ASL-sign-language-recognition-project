import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

with open('data_signs.pickle', 'rb') as f:
    dataset = pickle.load(f)

X = dataset['data']
y = dataset['labels']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir los datos de manera estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Modelo mejorado con Dropout y capa adicional
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(set(y_train)), activation='softmax')
])

# Optimización ajustada
model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Guardar el modelo
model.save('model.keras')