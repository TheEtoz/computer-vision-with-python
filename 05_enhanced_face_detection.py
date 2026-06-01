import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

print("Loading models... please wait...")

while True:
    ret, frame = cap.read()
    
    try:
        # Analyze the frame
        # 'actions' defines what we want to find
        results = DeepFace.analyze(frame, actions=['emotion', 'gender'], enforce_detection=False)
        
        for res in results:
            x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
            emotion = res['dominant_emotion']
            gender = res['dominant_gender']
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Put text
            text = f"{gender} | {emotion}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    except Exception as e:
        pass # Handle cases where no face is detected
    
    cv2.imshow('Enhanced Analysis', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()