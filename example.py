import cv2
import numpy as np
import fingerprint_feature_extractor
import math
from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform

import cv2
import numpy as np
import os
import skimage.morphology

def remove_background(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Створюємо маску для видалення фону
    mask = np.zeros(img.shape[:2], np.uint8)

    # Визначаємо область, що містить об'єкт
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)

    # Виконуємо GrabCut алгоритм для видалення фону
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Сегментуємо зображення на основі маски
    segmented = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Вирізаємо фон з оригінального зображення
    result = img * segmented[:, :, np.newaxis]

    return result

def skeletonize(img):
    # Конвертуємо зображення в градації сірого
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Бінаризуємо зображення за допомогою адаптивного порогування
    threshold = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Застосовуємо морфологічну операцію замикання
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    # Застосовуємо морфологічну операцію скелетизації
    skeleton = skimage.morphology.skeletonize(closing)

    # Конвертуємо скелет у формат BGR для відображення
    skeleton_bgr = cv2.cvtColor(skeleton.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    return skeleton_bgr


def align_orientation(image):
    rotation_angle = find_rotation_angle(image)
    print(rotation_angle)
    if rotation_angle < 45:
        aligned_angle = 0
    elif rotation_angle < 135:
        aligned_angle = 90
    elif rotation_angle < 225:
        aligned_angle = 90
    else:
        aligned_angle = 270

    # Виконуємо поворот зображення
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -aligned_angle, 1)
    aligned_image = cv2.warpAffine(image, M, (cols, rows))

    return aligned_image

def find_rotation_angle(image):
    # Обчислюємо градієнти за допомогою оператора Собеля
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Обчислюємо кут орієнтації для кожного пікселя
    orientation = np.arctan2(gradient_y, gradient_x)

    # Обчислюємо гістограму орієнтацій
    histogram, bins = np.histogram(orientation, bins=36, range=(-np.pi, np.pi))

    # Знаходимо найчастіший орієнтаційний інтервал
    max_interval = np.argmax(histogram)
    rotation_angle = (max_interval + 0.5) * 10

    return rotation_angle



if __name__ == '__main__':
    img1 = cv2.imread('enhanced/f1.jpeg', 1)
    img_without_bg = remove_background(img1)
    skeleton_img = skeletonize(img_without_bg)

    #cv2.imshow('Matched Image', matched_image)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img_without_bg)
    cv2.imshow('img3', skeleton_img)
    cv2.imwrite('./check_img.jpg', skeleton_img)
    # cv2.imshow('Result',  img_without_bg)
   # cv2.imshow('skeleton_img',  skeleton_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
