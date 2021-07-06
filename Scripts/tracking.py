import cv2
import numpy as np
import imutils

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('imgs/MVI_0846.avi')

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Error opening video  file")
   
first_flag = True
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == True:
        original = frame.copy()
        if first_flag:
            size = (frame.shape[0], frame.shape[1])
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3))*2,int(cap.get(4))))
            first_flag = False

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur frame
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform adaptive thresholding and invert so background is black
        thresh = 255 - cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Grab those contours
        cnts = imutils.grab_contours(cnts)

        # Loop through contours to filter and plot
        for c in cnts:

            # Compute moments of contour
            M = cv2.moments(c)

            # Compute contour perimeter
            peri = cv2.arcLength(c, True)

            # Find approximate polygon
            approx = cv2.approxPolyDP(c, 0.01*peri, True)

            # Calculate center of contours from moments
            mc = (M['m10'] / (M['m00'] + 1e-5), M['m01'] / (M['m00'] + 1e-5))

            # Filter by size and area since camera is always aligned on the machine
            #TODO: compute intersection over union and filter out diplicate contours
            if (mc[0] > 800) and (mc[0] < 1000) and (M['m00'] > 300):
                cv2.circle(frame, (int(mc[0]), int(mc[1])), 4, (255,0,0), -1)
                c = c.astype("int")
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 5)
            # Filter out coin for prototype sizing
            elif (len(approx) > 4) and (M['m00'] > 6000):
                cv2.circle(frame, (int(mc[0]), int(mc[1])), 4, (0,0,255), -1)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 5)
        stacked = np.concatenate((original, frame), axis=1)
        out.write(stacked)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# When everything done, release 
# the video capture object
out.release()
cap.release()
   
# Closes all the frames
cv2.destroyAllWindows()
waitKey(1)