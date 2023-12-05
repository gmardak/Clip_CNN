from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2
from training_model import VGG_CLIPS
#from gpiozero import Button
import RPi.GPIO as GPIO
from time import sleep

#input_signal = Button(16)
output_signal = 12
input_signal = 16
GPIO.setmode(GPIO.BCM)
GPIO.setup(output_signal, GPIO.OUT)
GPIO.setup(input_signal, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


def set_low():
    GPIO.output(output_signal, GPIO.LOW)
    
def set_high():
    GPIO.output(output_signal, GPIO.HIGH)

model = VGG_CLIPS()
model.load_model()
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
while True:
    #------------------------------------------------
    print('start waiting')
    while GPIO.input(input_signal) == GPIO.LOW:
        sleep(0.5)
    #------------------------------------------
    # Take a picture with the webcam
    frame = cf.capture_image_from_webcam_single()

    # Predicts the model
    prediction = model.predict()


    if prediction:
        set_high()
        sleep(0.5)
        set_low()
    else:
        set_high()
        sleep(2.5)
        set_low()
        
GPIO.cleanup()

    
