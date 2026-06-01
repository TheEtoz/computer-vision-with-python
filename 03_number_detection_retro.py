import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 1. Train the Model
print("Training model... please wait.")
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=1, verbose=0)
print("Model ready!")

layer_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.get_layer('conv1').output)

# 2. GUI Setup
img = np.zeros((280, 280), dtype=np.uint8)
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN: drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing: cv2.circle(img, (x, y), 15, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP: drawing = False

cv2.namedWindow('Draw Digit')
cv2.setMouseCallback('Draw Digit', draw)

print("--- Instructions: Press 'p' to see the computer's 'pipeline'. ---")

while True:
    cv2.imshow('Draw Digit', img)
    if cv2.waitKey(1) == ord('p'):
        # --- PIPELINE START ---
        # Stage 1: Resize
        small = cv2.resize(img, (28, 28))
        
        # Stage 2: Blur (Denoising)
        blurred = cv2.GaussianBlur(small, (3, 3), 0)
        
        # Stage 3: Threshold (Binarization)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        
        # Display Stages (Scaled up for visibility)
        cv2.imshow('1. Resized', cv2.resize(small, (280, 280), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('2. Gaussian Blur (Denoise)', cv2.resize(blurred, (280, 280), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('3. Final Threshold', cv2.resize(thresh, (280, 280), interpolation=cv2.INTER_NEAREST))
        
        # Feature Extraction
        input_data = thresh.reshape(1, 28, 28, 1).astype('float32') / 255.0
        features = layer_model.predict(input_data, verbose=0)
        cv2.imshow('4. CNN Feature Vision', cv2.resize(features[0,:,:,0], (280, 280), interpolation=cv2.INTER_NEAREST))
        
        print(f"Prediction: {np.argmax(model.predict(input_data, verbose=0))}")
        
    elif cv2.waitKey(1) == ord('c'):
        img[:] = 0
        for win in ['1. Resized', '2. Gaussian Blur (Denoise)', '3. Final Threshold', '4. CNN Feature Vision']:
            try: 
                cv2.destroyWindow(win)
            except: 
                pass
    elif cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()