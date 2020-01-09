#!/usr/bin/env python

import cv2
import numpy as np
import imutils

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('out.avi')

keyhit = True

lower = (35, 170, 110)
upper = (65, 255, 230)

def find_ball_center(img):
    lower = (35, 170, 110)
    upper = (65, 255, 230)

    resized = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))[1]
    else:
        return -100



while(cap.isOpened()):

    if keyhit:
        ret, frame = cap.read()

        if ret == True:
            cv2.imshow('Frame', frame)

            resized = cv2.resize(frame, (256, 256))
            cv2.imshow('Resized', resized)

            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            cv2.imshow('HSV', hsv)

            mask = cv2.inRange(hsv, lower, upper)
            cv2.imshow('Mask', mask)

            center = find_ball_center(frame)

            out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.circle(out, (130, center), 1, (0, 0, 255), -1)

            cv2.imshow('Mask w/ Center', out)

            keyhit = False

    key = cv2.waitKey(25)

    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('l'):
        keyhit = True

cap.release()
cv2.destroyAllWindows()
