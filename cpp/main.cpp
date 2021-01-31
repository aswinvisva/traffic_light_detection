#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "TrafficArrowDirection.cpp"
using namespace cv;
using namespace std;


int main(int argc, const char * argv[]) {
    Mat traffic_lights_1 = imread("/Users/aswinvisva/Desktop/traffic_light_detection/traffic_lights/1.jpg");
    
    findTrafficLights(traffic_lights_1, 301, 323, 126, 158);  // Red
    findTrafficLights(traffic_lights_1, 403, 413, 129, 162);  // Red
    
    Mat traffic_lights_2 = imread("/Users/aswinvisva/Desktop/traffic_light_detection/traffic_lights/2.jpg");
    
    findTrafficLights(traffic_lights_2, 235, 251, 36, 83);  // Red
    findTrafficLights(traffic_lights_2, 384, 401, 38, 85);  // Red
    
    Mat traffic_lights_3 = imread("/Users/aswinvisva/Desktop/traffic_light_detection/traffic_lights/3.jpg");
    
    findTrafficLights(traffic_lights_3, 257, 273, 67, 109);  // Green
    findTrafficLights(traffic_lights_3, 395, 411, 70, 110);  // Green
    
    Mat traffic_lights_4 = imread("/Users/aswinvisva/Desktop/traffic_light_detection/traffic_lights/4.jpg");
    
    findTrafficLights(traffic_lights_4, 283, 298, 82, 115);  // Red
    findTrafficLights(traffic_lights_4, 390, 404, 82, 117);  // Red
    
    Mat traffic_lights_5 = imread("/Users/aswinvisva/Desktop/traffic_light_detection/traffic_lights/5.jpg");
    
    findTrafficLights(traffic_lights_5, 323, 331, 126, 154);  // Green
    findTrafficLights(traffic_lights_5, 388, 396, 122, 149);  // Green
    
    Mat traffic_lights_6 = imread("/Users/aswinvisva/Desktop/traffic_light_detection/traffic_lights/7.jpg");
    
    findTrafficLights(traffic_lights_6, 291, 340, 106, 229);  // Yellow
    findTrafficLights(traffic_lights_6, 523, 569, 105, 205);  // Yellow
    
    Mat traffic_lights_7 = imread("/Users/aswinvisva/Desktop/traffic_light_detection/traffic_lights/8.jpg");
    
    findTrafficLights(traffic_lights_7, 258, 313, 49, 209);  // Yellow
    
    return 0;
}




