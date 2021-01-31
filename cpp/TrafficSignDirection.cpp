//
//  TrafficSignDirection.cpp
//  TrafficLightDetection
//
//  Created by Aswin Visva on 2020-02-20.
//  Copyright © 2020 Aswin Visva. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

class TrafficSignDirection {
    
private:
    static double angle( Point pt1, Point pt2, Point pt0 )
    {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
    }
    

    static void onmouse(int event, int x, int y, int flags, void* param)
    {
        cout << "x: " << x << ", y:" << y <<"\n";
    }

public:
    
    struct Direction
    {
    public:
        bool left, right, straight;
        
        Direction(bool left, bool right, bool straight) {
            this->left = left;
            this->right = right;
            this->straight = straight;
        }
    };
    
    static Direction GetDirection(Mat src, int trafficLightPosX, int trafficLighpPoxY, int width, int height) {
        
        const int MIN_DIRECTION_SIGN_SIZE = 50;
        const int NUMBER_OF_SIDES = 4;
        const double MIN_SIZE_RATIO = 0.85;
        const double MAX_SIZE_RATIO = 1.15;
        const int SIDE_ALLOWANCE = 1;
        const double MEAN_LEFT_RIGHT_DIFFERENCE_THRESHOLD = 0.075;
        const double MEAN_CENTER_THRESHOLD = 1.2;
        const int ROI_SIZE_FACTOR = 6;
        const int APPROX_POLY_EPSILON = 15;
        const int ADAPTIVE_THRESHOLD_BLOCK_SIZE = 21;
        const int ADAPTIVE_THRESHOLD_C = 0;
        const double THRESHOLD_VALUE = 100;
        const double MAX_VALUE = 255;
        
        const bool PERFORM_CHECKS = true;
        const bool DEBUG_STATE = false;
        
        Mat hsl, threshold, output, detected_edges, dst;
        
        adaptiveThreshold(src,output, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                          THRESH_BINARY,ADAPTIVE_THRESHOLD_BLOCK_SIZE,ADAPTIVE_THRESHOLD_C);

        inRange(output, Scalar(255, 255, 255), Scalar(255, 255, 255), threshold);
        
        blur(threshold, detected_edges, Size(1,1) );

        dst = Scalar::all(0);
        src.copyTo( dst, detected_edges);

        Rect traffic_light_roi;
        
        if (DEBUG_STATE) {
            imshow("Original", threshold);
            setMouseCallback("Original", onmouse, &src); // pass address of img here
        }
        

        traffic_light_roi.x = trafficLightPosX;
        traffic_light_roi.y = trafficLighpPoxY - (height/2);
        
        
        traffic_light_roi.width = width * ROI_SIZE_FACTOR;
        traffic_light_roi.height = width * ROI_SIZE_FACTOR;
        Mat crop = threshold(traffic_light_roi);
        Mat crop_copy = src(traffic_light_roi);
        
        if (DEBUG_STATE) {
            imshow("Crop", crop);
            waitKey(0);
        }

        std::vector<std::vector<cv::Point> > contours;
        cv::Mat contourOutput = crop.clone();
        cv::findContours(contourOutput, contours, RETR_LIST, CHAIN_APPROX_NONE );
        
        cv::Mat roiOutput;
        
        bool left = false, right = false, straight = false;
        
        //Get the traffic sign from the first ROI
        for (size_t idx = 0; idx < contours.size(); idx++) {
            vector<Point> approxvec;
            
            double area = contourArea(contours.at(idx));
            
            if (PERFORM_CHECKS) {
                //Remove noise by filtering by size
                if(area < MIN_DIRECTION_SIGN_SIZE) {
                    continue;
                }
            }
                        
            cv::Mat approx;
            approxPolyDP(contours[idx], approx, APPROX_POLY_EPSILON, true);
            
            int n = approx.checkVector(2);
            
            if (PERFORM_CHECKS) {
                //Ensure the contour is a quadrilateral
                if(n < NUMBER_OF_SIDES - SIDE_ALLOWANCE || n > NUMBER_OF_SIDES + SIDE_ALLOWANCE) {
                    continue;
                }
            }
                        
            Rect r = boundingRect(contours[idx]);
            int width = r.width;
            int height = r.height;
            
            if (PERFORM_CHECKS) {
                //Ensure the shape is roughly square
                if((height / width) < MIN_SIZE_RATIO || (height / width) > MAX_SIZE_RATIO) {
                    continue;
                }
            }
            
            Mat roi = crop_copy(r);
            Mat roi_binary;
            
            // Binary Threshold
            cv::threshold(roi,roi_binary, THRESHOLD_VALUE, MAX_VALUE, THRESH_BINARY);


            if (DEBUG_STATE) {
                imshow("ROI", roi_binary);
            }
            
            Size s = roi.size();
            
            //Create regions to determine the sign direction
            Rect left_rect(s.width/10,s.height/2.5,s.width/5,s.height/5);
            Rect right_rect(s.width - s.width/3,s.height/2.5,s.width/5,s.height/5);
            Rect center(0,s.height/6,s.width,s.height/5);
            
            rectangle(roi_binary,left_rect,Scalar(0,0,0),1,8,0);
            rectangle(roi_binary,right_rect,Scalar(0,0,0),1,8,0);
            rectangle(roi_binary,center,Scalar(0,0,0),1,8,0);
            
            Scalar mean_overall, dev_overall, mean, dev, mean_left, dev_left, mean_right, dev_right;
            
            meanStdDev(roi_binary(left_rect), mean_left, dev_left);
            meanStdDev(roi_binary(right_rect), mean_right, dev_right);
            meanStdDev(roi_binary(center), mean, dev);
            meanStdDev(roi_binary, mean_overall, dev_overall);
            
            double ratio_left = (double)mean_left[0] / (double)mean_overall[0];
            double ratio_right = (double)mean_right[0] / (double)mean_overall[0];
            double ratio_center = (double)mean[0] / (double)mean_overall[0];
            
            if(DEBUG_STATE) {
                cout << "Ratio left: " << ratio_left << "\n";
                cout << "Ratio right: " << ratio_right << "\n";
                cout << "Ratio straight: " << ratio_center << "\n";
                cout << "Left/Right ratio difference: " << abs(ratio_left-ratio_right) << "\n";
            }

            if(ratio_left < ratio_right && abs(ratio_left-ratio_right) > MEAN_LEFT_RIGHT_DIFFERENCE_THRESHOLD) {
                left = true;
            }
            else if(ratio_right < ratio_left && abs(ratio_left-ratio_right) > MEAN_LEFT_RIGHT_DIFFERENCE_THRESHOLD) {
                right = true;
            }
            
            if(ratio_center < MEAN_CENTER_THRESHOLD) {
                straight = true;
            }
            
            if (DEBUG_STATE) {
                imshow("Arrow Regions", roi_binary);
                waitKey(0);
            }
        
        }
        
        return *new Direction(left,right,straight);

    }
};
