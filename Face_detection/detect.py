import cv2 as cv
import numpy as np
cam=cv.VideoCapture(0)
detector=cv.CascadeClassifier ("haar_frontal.xml")

def main():
