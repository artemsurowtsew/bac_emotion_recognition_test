import numpy as np
from keras.models import model_from_json

# Завантаження архітектури моделі з JSON-файлу
json_file = open(r'C:\Users\Artem\PycharmProjects\pythonProject\emotion_model1505.json')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Завантаження ваг з H5-файлу
emotion_model.load_weights(r"C:\Users\Artem\PycharmProjects\pythonProject\emotion_model1505.h5")

print("Loaded model architecture and weights from disk")

# Припустимо, що у вас є прогнозовані значення (y_pred) та фактичні мітки (y_true)


y_pred = emotion_model.predict(X_test)  # Замініть X_test на ваші дані для тестування
mae = np.mean(np.abs(y_true - y_pred))

print(f"Mean Absolute Error (MAE): {mae:.4f}")