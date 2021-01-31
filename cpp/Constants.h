//
//  Constants.h
//  TrafficLightDetection
//
//  Created by Aswin Visva on 2020-10-11.
//  Copyright © 2020 Aswin Visva. All rights reserved.
//

#ifndef Constants_h
#define Constants_h

static const bool DEBUG_MODE = true;

static const double MIN_OBJECT_AREA = 2.0;

static const int GREEN_LOW_B = 160;
static const int GREEN_LOW_G = 160;
static const int GREEN_LOW_R = 0;
static const int GREEN_HI_B = 255;
static const int GREEN_HI_G = 255;
static const int GREEN_HI_R = 150;

static const int RED_LOW_B = 160;
static const int RED_LOW_G = 160;
static const int RED_LOW_R = 160;
static const int RED_HI_B = 255;
static const int RED_HI_G = 255;
static const int RED_HI_R = 255;

static const int YELLOW_LOW_B = 120;
static const int YELLOW_LOW_G = 160;
static const int YELLOW_LOW_R = 160;
static const int YELLOW_HI_B = 150;
static const int YELLOW_HI_G = 255;
static const int YELLOW_HI_R = 255;

static const int MIN_DIRECTION_SIGN_SIZE = 50;
static const int NUMBER_OF_SIDES = 4;
static const double MIN_SIZE_RATIO = 3;
static const double MAX_SIZE_RATIO = 10;
static const int SIDE_ALLOWANCE = 2;
static const double MEAN_LEFT_RIGHT_DIFFERENCE_THRESHOLD = 0.075;
static const double MEAN_CENTER_THRESHOLD = 1.2;
static const int ROI_SIZE_FACTOR = 6;
static const int APPROX_POLY_EPSILON = 15;
static const int ADAPTIVE_THRESHOLD_BLOCK_SIZE = 5;
static const int ADAPTIVE_THRESHOLD_C = 0;
static const double THRESHOLD_VALUE = 100;
static const double MAX_VALUE = 255;

#endif /* Constants_h */
