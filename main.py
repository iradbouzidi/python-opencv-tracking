import cv2
from vidgear.gears import CamGear

# Load some pre-trained data on car rear ends ( haar cascade algorithm )
car_tracker = cv2.CascadeClassifier('cars.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Get video footage

stream = CamGear(source='https://youtu.be/WriuvU1rXkc', y_tube=True,
                 time_delay=1, logging=True).start()

# Iterate forever over frames
while True:

    # Read the current frame
    frame = stream.read()
    # read frames

    if frame.any():
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    #cars = car_tracker.detectMultiScale(grayscaled_frame, scaleFactor=1.1, minNeighbors=2)

    # Detect pedestrians
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    # pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame, scaleFactor=1.1, minNeighbors=2)

    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+1), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with cars & pedestrians spotted
    cv2.namedWindow("Face Detector", 0)
    cv2.resizeWindow("Face Detector", 720, 480)
    cv2.imshow("Face Detector", frame)

    # Listen for a key press for 1ms, then move on
    key = cv2.waitKey(5)

    # Stop if Q is pressed
    if key == 81 or key == 113:
        break

# Close output window
cv2.destroyAllWindows()

stream.stop()
# Safely close video stream.

print("Code Complete")
