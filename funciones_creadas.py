import random
import cv2 as cv
import numpy as np
from statistics import *


def crop_coin_rm_bg(img, size):
    img = cv.resize(img, size)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (15, 15), 5)
    # detección de círculos
    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, int(0.3 * size[0]),
                              param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is None:
        return img
    circles = np.int16(np.around(circles))
    # eliminar círculos que se encuentran fuera de la imagen
    circles = circles.reshape(circles.shape[1], 3)
    circles = circles[np.logical_not(np.logical_or(circles[:, 0] + circles[:, 2] > size[0],
                                                   circles[:, 1] + circles[:, 2] > size[0]))]
    circles = circles[np.logical_not(np.logical_or(circles[:, 0] - circles[:, 2] < 0,
                                                   circles[:, 1] - circles[:, 2] < 0))]
    # encontrar radio mayor
    radios = circles[:, 2]
    radio_mayor = np.max(radios)
    index_radio_mayor = np.where(radios == radio_mayor)
    circle = circles[index_radio_mayor[0][0]]
    # cortar la imagen
    left_most = circle[0] - circle[2]
    top_most = circle[1] - circle[2]
    right_most = circle[0] + circle[2]
    bottom_most = circle[1] + circle[2]
    crop_img = img[top_most:bottom_most, left_most:right_most]
    # eliminar el fondo (en negro)
    black_mask = np.zeros(crop_img.shape[:2])
    cv.circle(black_mask, (radio_mayor, radio_mayor), radio_mayor, 255, -1)
    bytemask = np.asarray(black_mask, dtype=np.uint8)
    crop_img = cv.bitwise_not(crop_img, black_mask, bytemask)
    return crop_img


def compare(img, templates):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    scores = []
    for template in templates:
        template = cv.resize(template, (img.shape[0], img.shape[1]), interpolation=cv.INTER_AREA)
        template = cv.cvtColor(template, cv.COLOR_BGR2HSV)
        # comparación de las dos monedas ( o no monedas )
        score = cv.matchTemplate(img[0], template[0], cv.TM_CCORR_NORMED)
        scores.append(score[0][0].item())
    return mean(scores)


def get_scores(train_set, n):
    train_scores = []
    for c in range(len(train_set)):
        img = train_set[c]
        templates = random.sample(train_set[:c] + train_set[c + 1:], n)  # concatena sin el sol actual
        score = compare(img, templates)
        train_scores.append(score)
    return train_scores
