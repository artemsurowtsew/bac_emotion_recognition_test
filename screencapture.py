import time
import cv2
import mss
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the emotion detection model
json_file = open(r'C:\Users\Artem\PycharmProjects\pythonProject\model\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(r"C:\Users\Artem\PycharmProjects\pythonProject\model\emotion_model.h5")
print("Loaded model from disk")

# Function to preprocess image and perform emotion detection
def infer_emotion(image):
    # Preprocess the image to match the input size of the model
    input_image = cv2.resize(image, (48, 48))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    input_image = input_image.reshape((1, 48, 48, 1))  # Add batch dimension
    input_image = input_image / 255.0  # Normalize pixel values

    # Make predictions using the loaded model
    predictions = emotion_model.predict(input_image)

    # Process the predictions as needed
    # For example, you can return the predicted class:
    predicted_class = np.argmax(predictions)
    return predicted_class

cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)  # Create a named window

with mss.mss() as sct:
    monitor = {"top": 0, "left": 0, "width": 800, "height": 600}


    while "Screen capturing":
        last_time = time.time()

        img = np.array(sct.grab(monitor))

        # Perform emotion detection
        detected_emotion = infer_emotion(img)

        # Display the detected emotion on the existing window
        emotion_labels = ["Angry", "Happy", "Sad", "Surprised", "Neutral"]

        # Check if the index is within the valid range
        if 0 <= detected_emotion < len(emotion_labels):
            predicted_emotion = emotion_labels[detected_emotion]
            img = cv2.putText(img, f"Emotion: {predicted_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Emotion Detection", img)
        else:
            print(f"Detected Emotion Index is out of range: {detected_emotion}")

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
