import cv2
import numpy as np

def callback(x):
    pass

def hsv_mask():
    cv2.namedWindow('trackbars')
    # video = cv2.VideoCapture(0)

    # inital values
    ilowH = 16
    ihighH = 48
    ilowS = 147
    ihighS = 253
    ilowV = 150
    ihighV = 255

    # create trackbars for color change
    cv2.createTrackbar('lowH','trackbars',ilowH,180,callback)
    cv2.createTrackbar('highH','trackbars',ihighH,180,callback)

    cv2.createTrackbar('lowS','trackbars',ilowS,255,callback)
    cv2.createTrackbar('highS','trackbars',ihighS,255,callback)

    cv2.createTrackbar('lowV','trackbars',ilowV,255,callback)
    cv2.createTrackbar('highV','trackbars',ihighV,255,callback)

    while True:
        # ret, frame = video.read()
        frame = cv2.imread("DSC_0089.JPG")
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ilowH = cv2.getTrackbarPos('lowH', 'trackbars')
        ihighH = cv2.getTrackbarPos('highH', 'trackbars')
        ilowS = cv2.getTrackbarPos('lowS', 'trackbars')
        ihighS = cv2.getTrackbarPos('highS', 'trackbars')
        ilowV = cv2.getTrackbarPos('lowV', 'trackbars')
        ihighV = cv2.getTrackbarPos('highV', 'trackbars')
        lower_hsv = np.array([ilowH, ilowS, ilowV])
        higher_hsv = np.array([ihighH, ihighS, ihighV])
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
        dim = (width//2, height//2)
        cv2.imshow('trackbars', cv2.resize(mask,dim, interpolation = cv2.INTER_AREA))
        cv2.imshow('original image', cv2.resize(frame,dim, interpolation = cv2.INTER_AREA))
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()
    return lower_hsv, higher_hsv

print(hsv_mask())
