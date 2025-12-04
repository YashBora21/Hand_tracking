import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
prev_time = 0

# Danger Zone (Virtual Object)
obj_x1, obj_y1 = 400, 100
obj_x2, obj_y2 = 600, 300

WARNING_THRESHOLD = 120
DANGER_THRESHOLD = 55

def nothing(x): pass

# Trackbars for skin calibration
cv2.namedWindow("Skin Controls")
cv2.createTrackbar("L-H", "Skin Controls", 0, 179, nothing)
cv2.createTrackbar("L-S", "Skin Controls", 40, 255, nothing)
cv2.createTrackbar("L-V", "Skin Controls", 60, 255, nothing)
cv2.createTrackbar("U-H", "Skin Controls", 25, 179, nothing)
cv2.createTrackbar("U-S", "Skin Controls", 200, 255, nothing)
cv2.createTrackbar("U-V", "Skin Controls", 255, 255, nothing)

# Fingertip detection using convex hull highest peak
def detect_fingertip(contour):
    hull = cv2.convexHull(contour)
    if hull is None or len(hull) < 3:
        return None
    topmost = tuple(hull[hull[:,:,1].argmin()][0])
    return topmost

def distance_to_rect(px, py):
    nearest_x = np.clip(px, obj_x1, obj_x2)
    nearest_y = np.clip(py, obj_y1, obj_y2)
    return np.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Trackbar HSV inputs
    lh = cv2.getTrackbarPos("L-H", "Skin Controls")
    ls = cv2.getTrackbarPos("L-S", "Skin Controls")
    lv = cv2.getTrackbarPos("L-V", "Skin Controls")
    uh = cv2.getTrackbarPos("U-H", "Skin Controls")
    us = cv2.getTrackbarPos("U-S", "Skin Controls")
    uv = cv2.getTrackbarPos("U-V", "Skin Controls")

    lower_skin = np.array([lh, ls, lv], dtype=np.uint8)
    upper_skin = np.array([uh, us, uv], dtype=np.uint8)

    # Full mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.dilate(mask, None, 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    state = "SAFE"
    distance = None

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # Reject face/torso based on location/area
        x, y, w, h = cv2.boundingRect(cnt)
        if 2000 < area < 35000 and y > 80:
            fingertip = detect_fingertip(cnt)

            if fingertip:
                cx, cy = fingertip
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

                distance = distance_to_rect(cx, cy)

                if distance <= DANGER_THRESHOLD:
                    state = "DANGER"
                elif distance <= WARNING_THRESHOLD:
                    state = "WARNING"

            cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)

    # Color for danger zone based on state
    color = (0,255,0) if state == "SAFE" else (0,255,255) if state == "WARNING" else (0,0,255)
    cv2.rectangle(frame, (obj_x1, obj_y1), (obj_x2, obj_y2), color, 3)

    cv2.putText(frame, f"State: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    if state == "DANGER":
        cv2.putText(frame, "DANGER DANGER!", (100, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,0,255), 5)

    if distance:
        cv2.putText(frame, f"Dist: {int(distance)} px", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # FPS Display
    curr_time = time.time()
    fps = 1/(curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS:{int(fps)}", (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Hand Danger System", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
