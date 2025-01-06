import cv2
import time
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


width_cam, height_cam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Pycaw initialization for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_vol, max_vol = volume_range[0], volume_range[1]  # Volume range

pTime = 0
vol = 0
vol_bar = 400
vol_per = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame!")
        break

    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Process hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions for index finger tip and thumb tip
            h, w, c = img.shape
            x1, y1 = int(hand_lms.landmark[8].x * w), int(hand_lms.landmark[8].y * h)  # Index finger tip
            x2, y2 = int(hand_lms.landmark[4].x * w), int(hand_lms.landmark[4].y * h)  # Thumb tip

            # Draw circles and line between the two points
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Calculate the distance between the two points
            length = np.hypot(x2 - x1, y2 - y1)

            # Map the distance to the volume range
            vol = np.interp(length, [20, 200], [min_vol, max_vol])  
            vol_bar = np.interp(length, [20, 200], [400, 150])
            vol_per = np.interp(length, [20, 200], [0, 100])
            volume.SetMasterVolumeLevel(vol, None)

            # Interpolate color from green to red based on volume percentage
            r = int(np.interp(vol_per, [0, 100], [255, 0]))  # Red increases as volume decreases
            g = int(np.interp(vol_per, [0, 100], [0, 255]))  # Green decreases as volume decreases
            color = (0, g, r)

           
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), color, cv2.FILLED)
            cv2.putText(img, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, color, 3)

    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    
    cv2.imshow("Hand Tracking Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
