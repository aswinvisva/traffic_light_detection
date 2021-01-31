#include <iostream>
#include <assert.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xphoto/white_balance.hpp>

using namespace cv;
using namespace std;

struct TrafficLight
{
public:
    int state; // 0 -> Green, 1 -> Yellow, 2 -> Red
    int direction; // -1 -> None, 0 -> Left, 1 -> Right
    
    TrafficLight(int state, int direction) {
        assert (state == 0 || state == 1 || state == 2);
        this->state = state;
        this->direction = direction;
    }
};


static int getArrowDirection(Mat arrowROI) {
    /** @brief Gets arrow direction given a region-of-interest around the traffic light arrow

    @param arrowROI Input one-channel grayscale image
    @return {0 -> Left, 1 -> Right}
    */
    
    // Split image into two halves
    Mat halfLeft = arrowROI(Rect(0, 0, arrowROI.cols/2, arrowROI.rows));
    Mat halfRight = arrowROI(Rect(arrowROI.cols/2, 0, arrowROI.cols/2, arrowROI.rows));
    
    // Get non-zero pixel counts on both halves
    int leftCount = countNonZero(halfLeft);
    int rightCount = countNonZero(halfRight);
    
    // Get direction by comparing pixel counts
    return (int)leftCount > rightCount;
}

static int arrowDetection(Mat roi) {
    /** @brief Arrow detection and classification from a traffic light region-of-interest

    @param bgrROI Input three-channel BGR image of traffic light ROI
    @return {-1 -> No traffic arrow, 0 -> Left, 1 -> Right}
    */
        
    // Apply blur
    Mat blurred;
    cv::blur(roi, blurred, cv::Size(1, 1));
    
    Mat morph;
    Mat element = getStructuringElement(0, Size(1, 1));
    
    // Apply dilation then erosion with morph close
    cv::morphologyEx(blurred, morph, MORPH_CLOSE, element);
    
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = morph.clone();
    
    // Find contours
    cv::findContours(contourOutput, contours, RETR_LIST, CHAIN_APPROX_NONE);
    vector<Point> approx;
    cv::Mat res = cv::Mat::zeros(roi.size(), CV_8UC3);

    for( size_t i = 0; i < contours.size(); i++ ) {
        double area = contourArea(contours[i]);
        
        if(area <= 30) continue; // Filter area
        
        approxPolyDP(contours[i], approx, 10.0, true);
        
        Rect br = boundingRect(contours[i]);
        
        // Get aspect ratio of contour
        double aspectRatio = double(br.width) / double(br.height);
        
        cout << aspectRatio << ", " << approx.size() << "\n";
        
        Mat arrowROI = contourOutput(br);
                        
        if(approx.size() == 3 && abs(aspectRatio - 1) < 0.75) { // Filter by aspect ratio and by number of edges
            Mat arrowROI = contourOutput(br);
            
            imshow("grayArrow", arrowROI);
            waitKey(0);
            
            return getArrowDirection(arrowROI);
        }
    }
    
    return -1;
}

static Mat bulbDetection(Mat roi) {
    /** @brief Bulb detection and classification from a traffic light region-of-interest

    @param roi Input one-channel grayscale image of traffic light ROI
    */
    
    Mat blurred;
    
    // Apply blur
    cv::blur(roi, blurred, cv::Size(1, 1));
    
    Mat morph;
    Mat element = getStructuringElement(0, Size(1, 1));
    
    // Apply dilation then erosion with morph close
    cv::morphologyEx(blurred, morph, MORPH_CLOSE, element);
    
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = morph.clone();
    
    // Find contours
    cv::findContours(contourOutput, contours, RETR_LIST, CHAIN_APPROX_NONE);
    vector<Point> approx;
    cv::Mat res = cv::Mat::zeros(roi.size(), CV_8UC3);

    for( size_t i = 0; i < contours.size(); i++ ) {
        double area = contourArea(contours[i]);
        
        if(area <= 30) continue;
        
        approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.02, true);
        
        Rect br = boundingRect(contours[i]);
        
        double aspectRatio = double(br.width) / double(br.height);
        
        if(approx.size() >= 4 && abs(aspectRatio - 1) < 0.75) {
            drawContours(res, contours, (int)i, Scalar(255, 255, 255), FILLED);
        }
    }
    
    Mat grayRes;
    
    cvtColor(res, grayRes, COLOR_BGR2GRAY);
    
    return grayRes;
}


static TrafficLight findTrafficLights(Mat img, int x1, int x2, int y1, int y2) {
    /** @brief Get traffic bulbs and arrows

    @param img Input three-channel BGR image
    @param x1 left-most coordinate of traffic light
    @param x2 right-most coordinate of traffic light
    @param y1 bottom coordinate of traffic light
    @param y2 top coordinate of traffic light
    */
    
    Mat original = img;
    
    Mat res;
    
    // White balancing
    Ptr<xphoto::WhiteBalancer> wb = cv::xphoto::createGrayworldWB();
    wb->balanceWhite(img, res);
    
    Mat roi = res(Rect(x1, y1, x2-x1, y2-y1)); // Get ROI where traffic lights are likely to be
    
    Mat hsv;
    cv::cvtColor(roi, hsv, COLOR_BGR2HSV);
    
    Mat greenBlobs;
    Mat redBlobs1;
    Mat redBlobs2;
    Mat yellowBlobs;
    
    // Get red, yellow and green bulbs
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), redBlobs1);
    inRange(hsv, Scalar(160, 175, 100), Scalar(180, 255, 255), redBlobs2);
    inRange(hsv, Scalar(40, 20, 100), Scalar(90, 255, 255), greenBlobs);
    inRange(hsv, Scalar(10, 30, 225), Scalar(40, 255, 255), yellowBlobs);
    
    Mat maskR, maskG, maskY;
    Mat maskRBlur, maskGblur, maskYBlur;
    
    // Apply blur
    cv::add(redBlobs1, redBlobs2, maskR);
    cv::blur(maskR, maskRBlur, cv::Size(2, 2));
    cv::blur(greenBlobs, maskGblur, cv::Size(2, 2));
    cv::blur(yellowBlobs, maskYBlur, cv::Size(2, 2));
    
    // Find bulbs
    Mat yBulbs = bulbDetection(maskYBlur);
    Mat gBulbs = bulbDetection(maskGblur);
    Mat rBulbs = bulbDetection(maskRBlur);

    int arrowY = arrowDetection(maskYBlur);
    int arrowG = arrowDetection(maskGblur);
    int arrowR = arrowDetection(maskRBlur);
    
    int arrowDirection = max(arrowY, arrowG);
    arrowDirection = max(arrowDirection, arrowR);
    
    cout << arrowY << "\n";
    cout << arrowG << "\n";
    cout << arrowR << "\n";

    // Find state by comparing pixel counts
    int y_pixels = countNonZero(yBulbs);
    int g_pixels = countNonZero(gBulbs);
    int r_pixels = countNonZero(rBulbs);
    
//    cout << arrowa;

    if (y_pixels > g_pixels && y_pixels > r_pixels) {
        cout << "Yellow\n";
        return TrafficLight(1, arrowDirection);
    }
    else if (g_pixels > y_pixels && g_pixels > r_pixels) {
        cout << "Green\n";
        return TrafficLight(0, arrowDirection);
    }
    else {
        cout << "Red\n";
        return TrafficLight(2, arrowDirection);
    }
}

