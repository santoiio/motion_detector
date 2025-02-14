import cv2, time, pandas
from datetime import datetime

first_frame = None # Store the reference background frame
status_list = [None, None]
times = [] # Store timestamps when motion starts and stops
df = pandas.DataFrame(columns=["Start", "End"])

# Initialize webcam and save motion as mp4 file
video = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output1.mp4', fourcc, 33.0, (640, 480))

while True:
    check, frame = video.read()
    # Make sure frame is valid before continuing
    if not check or frame is None:
        print("Warning: Frame not captured. Skipping...")
        continue

    status = 0 # 0 means no motion, 1 means motion

    # Convert frame to grayscale and blur for easier motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 1)

    # Initialize first_frame or update it gradually
    if first_frame is None:
        first_frame = gray.copy()
        continue
    else:
        first_frame = cv2.addWeighted(first_frame, 0.95, gray,
                                        0.05, 0)

    # Compute absolute difference between current frame and background
    delta_frame = cv2.absdiff(first_frame, gray)
    # Highlight changes with threshold
    thresh_frame = cv2.threshold(delta_frame, 80, 255,
                                 cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # If object is large enough, motion is recorded
    (contours, _)  =  cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Bigger number means less sensitivity
        if cv2.contourArea(contour) < 500:
            continue
        status = 1 # Motion detected

        # Draw a bounding box around detected motion
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x +w , y + h), (128, 254, 0), 3)

    status_list.append(status)
    status_list = status_list[-2:]

    # Store timestamps when motion starts or stops
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # Save motion frames to video
    if status == 1:
        out.write(frame)

    # Display windows of each process at work
    cv2.imshow("Gray", gray)
    cv2.imshow("Delta", delta_frame)
    cv2.imshow("Threshold", thresh_frame)
    cv2.imshow("Color", frame)

    key = cv2.waitKey(33)

    # Press 'q' to exit and save
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

# Save motion timestamps to CSV
motion_data = [{"Start": times[i], "End": times[i+1]}
                for i in range(0, len(times) - 1, 2)]
df = pandas.DataFrame(motion_data)
df.to_csv("Times.csv")

# Release resources and close windows
video.release()
out.release()
cv2.destroyAllWindows()
