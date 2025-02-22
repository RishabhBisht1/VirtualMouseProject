import cv2
import numpy as np
import HandTrackingModule as htm
import time
from pynput.mouse import Controller, Button
from screeninfo import get_monitors

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, wCam)
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.handDetector(maxHands=1)
monitor = get_monitors()[0]  # Get primary monitor
wScreen, hScreen = monitor.width, monitor.height
# print(wScreen, hScreen)

mouse = Controller()
frameRed = 100  # frame reduction
smoothening = 15
while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print(x1,y1,x2,y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        cv2.rectangle(img, (frameRed, frameRed), (wCam - frameRed, hCam - frameRed), (255, 0, 255), 2)
        # 4. Only Index Finger : Moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameRed, wCam - frameRed), (0, wScreen))
            y3 = np.interp(y1, (frameRed, hCam - frameRed), (0,hScreen))

            # 6. Smoothen Values ( to reduce shaking of cursor)
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            mouse.position = (wScreen - clocX, clocY)  # when we move right mouse also moves right
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both Index and middle fingers are up : Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find Distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            # 10. Click mouse if distance shorten
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                mouse.click(Button.left, 1)

    # 11. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # 12. Display

    cv2.imshow("Image", img)
    cv2.waitKey(1)
