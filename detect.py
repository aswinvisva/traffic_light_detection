import math
from math import degrees, atan2, pi

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.path as mplPath


def normalize_image(img):
    V = np.mean(img[:, :, 2])
    kV = 10
    stdV = np.std(img[:, :, 2])
    stdV_m = np.std(img[:, :, 2] - V)

    img[:, :, 2] = kV * V + stdV * (img[:, :, 2] - V)
    return img


def weighted_hue(img):
    green_h = 90
    yellow_h = 20
    red_h = 0

    green_w = np.exp(-1 * np.square((img[:, :, 0] - green_h)) / (np.std(np.square((img[:, :, 0] - green_h)))))
    red_w = np.exp(-1 * np.square((img[:, :, 0] - red_h)) / (np.std(np.square((img[:, :, 0] - red_h)))))
    yellow_w = np.exp(-1 * np.square((img[:, :, 0] - yellow_h)) / (np.std(np.square((img[:, :, 0] - yellow_h)))))
    weight = np.maximum(green_w, red_w)
    weight = np.maximum(weight, yellow_w)

    img[:, :, 1] = np.multiply(weight, img[:, :, 1])

    return img


def arrow_detection(mask, debug=False):
    img = cv.imread('template1.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    sift = cv.xfeatures2d.SURF_create()
    kp1, des1 = sift.detectAndCompute(gray, None)
    kp2, des2 = sift.detectAndCompute(mask, None)

    print(des1)

    # create BFMatcher object
    # Match descriptors.
    # Create BFMatcher and add cluster of training images. One for now.
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)  # crossCheck not supported by BFMatcher
    clusters = np.array([des1])
    bf.add(clusters)

    # Train: Does nothing for BruteForceMatcher though.
    bf.train()

    matches = bf.match(des2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv.drawMatches(gray, kp1, mask, kp2, matches, img, (255, 255, 255))
    cv.imshow("ASD", img3)
    cv.waitKey(0)


def bulb_detection(mask, debug=False):
    mask = cv.blur(mask, (1, 1))

    if debug:
        cv.imshow('Bilateral', mask)
        cv.waitKey(0)

    kernel = np.ones((1, 1), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    if debug:
        cv.imshow('Morph', mask)
        cv.waitKey(0)

    im2, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    test_img = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    contour_image = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)

    if debug:
        cv.imshow('Contours', cv.drawContours(contour_image, contours, -1, (255, 255, 255)))
        cv.waitKey(0)

    print("=" * 8)

    for c in contours:
        approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
        area = cv.contourArea(c)
        x, y, w, h = cv.boundingRect(c)
        aspect_ratio = float(w) / h

        print("Ratio: %f, Area %f, Sides: %d" % (aspect_ratio, area, len(approx)))

        if (len(approx) >= 4) & (area > 30) & (abs(aspect_ratio - 1) < 0.75):
            cv.drawContours(test_img, [c], 0, (255, 255, 255), cv.FILLED)

    if debug:
        cv.imshow("Bulbs", test_img)
        cv.waitKey(0)

    return cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)


def detect_traffic_lights(img, x, y, x2, y2, img_name="test", ground_truth_class="Green", debug=False):
    original = img

    wb = cv.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    img = wb.balanceWhite(img)

    img = img[y:y2, x:x2]
    crop = img

    if debug:
        cv.imshow("original", img)
        cv.waitKey(0)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # color range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 175, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([40, 20, 100])
    upper_green = np.array([90, 255, 255])
    lower_yellow = np.array([10, 30, 225])
    upper_yellow = np.array([40, 255, 255])
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    maskg = cv.inRange(hsv, lower_green, upper_green)
    masky = cv.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv.add(mask1, mask2)
    maskg = cv.blur(maskg, (2, 2))
    masky = cv.blur(masky, (2, 2))
    maskr = cv.blur(maskr, (2, 2))

    if debug:
        cv.imshow('GMask', maskg)
        cv.imshow('RMask', maskr)
        cv.imshow('YMask', masky)
        cv.waitKey(0)

    arrow_detection(img, debug=debug)

    y_bulbs = bulb_detection(masky, debug=debug)
    g_bulbs = bulb_detection(maskg, debug=debug)
    r_bulbs = bulb_detection(maskr, debug=debug)

    all_bulbs = np.zeros(y_bulbs.shape, np.uint8)
    all_bulbs += y_bulbs
    all_bulbs += g_bulbs
    all_bulbs += r_bulbs

    if debug:
        cv.imshow('asd', all_bulbs)
        cv.waitKey(0)

    y_pixels = cv.countNonZero(y_bulbs)
    g_pixels = cv.countNonZero(g_bulbs)
    r_pixels = cv.countNonZero(r_bulbs)

    names = ["Red", "Yellow", "Green"]
    vals = [r_pixels, y_pixels, g_pixels]

    class_idx = np.argmax(vals)
    print("Classification: %s" % names[class_idx])

    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2

    original = cv.putText(original, 'Ground Truth: %s' % ground_truth_class, (0, 30), font,
                          fontScale, (0, 255, 0), thickness, cv.LINE_AA)
    original = cv.putText(original, 'Prediction: %s' % names[class_idx], (0, 60), font,
                          fontScale, (0, 255, 0), thickness, cv.LINE_AA)

    cv.imwrite("results/%s_results.png" % img_name, all_bulbs)
    cv.imwrite("results/%s_original.png" % img_name, original)
    cv.imwrite("results/%s_crop.png" % img_name, crop)

    f = open("results/%s_results.txt" % img_name, "w")
    f.write("\nPredicted Class: %s" % names[class_idx])
    f.write("\nTrue Class: %s" % ground_truth_class)
    f.close()


if __name__ == '__main__':
    # traffic_lights/1.jpg
    # Class: Red
    detect_traffic_lights(cv.imread("traffic_lights/1.jpg"), x=301, y=126, x2=323, y2=158,
                          img_name="traffic_lights_1_1", ground_truth_class="Red")
    detect_traffic_lights(cv.imread("traffic_lights/1.jpg"), x=403, y=129, x2=413, y2=162,
                          img_name="traffic_lights_1_2", ground_truth_class="Red")

    # traffic_lights/2.jpg
    # Class: Red
    detect_traffic_lights(cv.imread("traffic_lights/2.jpg"), x=235, y=36, x2=251, y2=83,
                          img_name="traffic_lights_2_1", ground_truth_class="Red")
    detect_traffic_lights(cv.imread("traffic_lights/2.jpg"), x=384, y=38, x2=401, y2=85,
                          img_name="traffic_lights_2_2", ground_truth_class="Red")

    # traffic_lights/3.jpg
    # Class: Green
    detect_traffic_lights(cv.imread("traffic_lights/3.jpg"), x=257, y=67, x2=273, y2=109,
                          img_name="traffic_lights_3_1", ground_truth_class="Green")
    detect_traffic_lights(cv.imread("traffic_lights/3.jpg"), x=395, y=70, x2=411, y2=110,
                          img_name="traffic_lights_3_2", ground_truth_class="Green")

    # traffic_lights/4.jpg
    # Class: Red
    detect_traffic_lights(cv.imread("traffic_lights/4.jpg"), x=283, y=82, x2=298, y2=115,
                          img_name="traffic_lights_4_1", ground_truth_class="Red")
    detect_traffic_lights(cv.imread("traffic_lights/4.jpg"), x=390, y=82, x2=404, y2=117,
                          img_name="traffic_lights_4_2", ground_truth_class="Red")

    # traffic_lights/5.jpg
    # Class: Green
    detect_traffic_lights(cv.imread("traffic_lights/5.jpg"), x=323, y=126, x2=331, y2=154,
                          img_name="traffic_lights_5_1", ground_truth_class="Green")
    detect_traffic_lights(cv.imread("traffic_lights/5.jpg"), x=388, y=122, x2=396, y2=149,
                          img_name="traffic_lights_5_2", ground_truth_class="Green")

    # traffic_lights/7.jpg
    # Class: Yellow

    detect_traffic_lights(cv.imread("traffic_lights/7.jpg"), x=291, y=106, x2=340, y2=229,
                          img_name="traffic_lights_7_1", ground_truth_class="Yellow")
    detect_traffic_lights(cv.imread("traffic_lights/7.jpg"), x=523, y=105, x2=569, y2=205,
                          img_name="traffic_lights_7_2", ground_truth_class="Yellow")
