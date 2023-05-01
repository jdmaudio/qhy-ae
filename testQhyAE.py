"""
Auto Exposure and Gain Control

This script captures video using pysky360.QHYCamera and automatically adjusts exposure and gain settings to achieve a desired mean sample value (MSV) for the captured frames. The algorithm uses proportional-integral (PI) control based on the image histogram. 
When decreasing MSV, gain is prioritized to be reduced before adjusting exposure.

Functions:

calculate_brightness: Calculates the mean brightness of a frame.
adjust_exposure: Adjusts exposure and gain settings based on the target MSV, maximum exposure value, and minimum gain value.
The main loop captures video frames, calculates brightness, and adjusts exposure and gain settings accordingly. 

Reference: https://github.com/alexzzhu/auto_exposure_control
"""

import cv2
import pysky360
import numpy as np


def calculate_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness


# Auto adjusts to mean brightness level over frame or ROI
def adjust_exposure(cv_image, current_exposure=1000, current_gain=20, targetMSV=2.2, max_exposure=50000, min_gain=15):
    (rows, cols, channels) = cv_image.shape
    if channels == 3:
        brightness_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)[:, :, 2]
    else:
        brightness_image = cv_image

    # crop_size = 500
    # rows//=2
    # cols//=3
    # brightness_image = brightness_image[rows-crop_size:rows+crop_size, cols-crop_size:cols+crop_size]
    # (rows, cols) = brightness_image.shape
    # cv2.imshow('brightness_image', brightness_image)

    hist = cv2.calcHist([brightness_image], [0], None, [5], [0, 256])

    mean_sample_value = 0
    for i in range(len(hist)):
        mean_sample_value += hist[i] * (i + 1)

    mean_sample_value /= (rows * cols)

    # Proportional and integral constants (k_p and k_i) - increasing will make exposure adjustments more aggressive
    k_p = 1600 
    k_i = 320
    max_i = 3

    err_p = targetMSV - mean_sample_value
    global err_i
    err_i += err_p

    if abs(err_i) > max_i:
        err_i = np.sign(err_i) * max_i

    if abs(err_p) > 0.5:
        if err_p < 0 and current_gain > min_gain:
            gain_decrement = abs(err_p) * 0.5
            new_gain = max(current_gain - gain_decrement, min_gain)
            camera.setControl(pysky360.ControlParam.Gain, new_gain, False)
            return current_exposure, new_gain

        new_exposure = current_exposure + k_p * err_p + k_i * err_i

        if new_exposure > max_exposure:
            new_exposure = max_exposure
            gain_increment = (targetMSV - mean_sample_value) * 0.75
            new_gain = current_gain + gain_increment
            camera.setControl(pysky360.ControlParam.Gain, new_gain, False)
            return new_exposure, new_gain

        camera.setControl(pysky360.ControlParam.Exposure, new_exposure, False)
        return new_exposure, current_gain
    else:
        return current_exposure, current_gain

    
current_exposure = 1000.0  # Set the initial exposure value (day)
# current_exposure = 90000.0 (night)
current_gain = 20  # Set the initial gain value
err_i = 0

camera = pysky360.QHYCamera()

cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL);
cv2.resizeWindow("Live Video", (800, 800));

camera.open('')
camera.setControl(pysky360.ControlParam.Exposure, current_exposure, False)
camera.setControl(pysky360.ControlParam.Gain, current_gain, False)
camera.setControl(pysky360.ControlParam.TransferBits, 8, False)

wait_frames = 0  # Number of frames to wait after setting a new exposure value is set
wait_counter = 0  # Counter to keep track of waited frames

while True:
    frame = camera.getFrame(True)
    current_brightness = calculate_brightness(frame)
    print(f"Current Brightness: {current_brightness}")

    # Only adjust exposure if the wait_counter is zero
    if wait_counter == 0:

        current_exposure, current_gain = adjust_exposure(frame, current_exposure, current_gain, 1.8)
        print(f"Current exposure: {current_exposure}, Current gain: {current_gain}")
        wait_counter = wait_frames
    else:
        wait_counter -= 1

    cv2.imshow('Live Video', frame)

    keyPressed = cv2.waitKey(10) & 0xFF
    if keyPressed == 27:
        break

cv2.destroyAllWindows()
