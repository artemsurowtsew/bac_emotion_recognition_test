import time
import cv2
import mss
import numpy as np
from tensorflow.keras.models import model_from_json
from dlib import dlib_detector

# Load the emotion detection model
json_file = open(r'C:\Users\Artem\PycharmProjects\pythonProject\model\surovtsev4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(r"C:\Users\Artem\PycharmProjects\pythonProject\model\surovtsev4.h5")
print("Loaded model from disk")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier( r'C:\Users\Artem\PycharmProjects\pythonProject\haarcascades/haarcascade_frontalface_default.xml')


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
    monitor = {"top": 0, "left": 0, "width": 1000, "height": 1000}

    while "Screen capturing":
        last_time = time.time()


        img = np.array(sct.grab(monitor))

        # Convert the image to grayscale for face detection
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dlib_faces = dlib_detector(gray_img)
        # Perform face detection
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop face region for emotion detection
            face_roi = img[y:y + h, x:x + w]

            # Perform emotion detection on the cropped face
            detected_emotion = infer_emotion(face_roi)



            # Display the detected emotion on the existing window
            emotion_labels = ["Angry","Disgusted","Fearful", "Happy","Neutral", "Sad", "Surprised"]

            if 0 <= detected_emotion < len(emotion_labels):
                predicted_emotion = emotion_labels[detected_emotion]
                cv2.putText(img, f"Emotion: {predicted_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 1, cv2.LINE_AA)
            else:
                print(f"Detected Emotion Index is out of range: {detected_emotion}")
        cv2.imshow("Emotion Detection", img)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
