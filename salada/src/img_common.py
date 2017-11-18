# -*- coding: utf-8 -*- 
import os
import sys
import cv2

def resize_image(image, size):
    """

    :param image: np.array
    :param size: tuple(x, y)
    :return: np.array
    """
    resize_img = cv2.resize(image, size)
    return resize_img

def display_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, save_name):
    cv2.imwrite(image, save_name)