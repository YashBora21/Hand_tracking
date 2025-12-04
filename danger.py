import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
prev_time = 0

# Virtual object rectangle
obj_x1, obj_y1 = 400, 100
obj_x2, obj_y2 = 600, 300

# Distance thresholds (tune if needed)
WARNING_THRESHOLD = 120
DANGER_THRESHOLD = 50

# Skin HSV range (adjustable)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)


def get_hand_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 1500:  # ignore small blobs
        return None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), c


def point_to_rect_distance(px, py, x1, y1, x2, y2):
    nearest_x = np.clip(px, x1, x2)
    nearest_y = np.clip(py, y1, y2)
    return np.sqrt((px - nearest_x)**2 + (py - nearest_y)**2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin segmentation
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    state = "SAFE"
    distance = None

    result = get_hand_center(mask)
    if result:
        (cx, cy), contour = result

        distance = point_to_rect_distance(cx, cy, obj_x1, obj_y1, obj_x2, obj_y2)

        if distance <= DANGER_THRESHOLD:
            state = "DANGER"
        elif distance <= WARNING_THRESHOLD:
            state = "WARNING"
        else:
            state = "SAFE"

        cv2.drawContours(frame, [contour], -1, (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 8, (0,255,0), -1)

    # Change rectangle border color based on state
    if state == "SAFE":
        color = (0,255,0)
    elif state == "WARNING":
        color = (0,255,255)
    else:
        color = (0,0,255)

    cv2.rectangle(frame, (obj_x1, obj_y1), (obj_x2, obj_y2), color, 3)

    # Overlay Texts
    cv2.putText(frame, f"State: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if state == "DANGER":
        cv2.putText(frame, "DANGER DANGER", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)

    # Show distance debug
    if distance is not None:
        cv2.putText(frame, f"Dist: {int(distance)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (500, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Danger Detection", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
