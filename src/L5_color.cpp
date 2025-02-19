#include <opencv2/opencv.hpp>
#include "./reusables/utils.h"
#include <iostream>
#include <cmath>

#define EPSILON 1.0e-5

double calcH(double r, double g, double b) {
    double r_g = r - g;
    double r_b = r - b;
    double g_b = g - b;

    double num = 0.5F * (r_g + r_b);
    double den = std::sqrt((r_g * r_g) + (r_b * g_b));

    double theta = std::acos(num / (den + EPSILON));

    double h;
    if (b <= g) {
        h = theta;
    }
    else {
        h = (CV_PI * 2) - theta;
    }

    return (h * CV_PI) / 180;
}

double calcS(double r, double g, double b) {
    return 1.0 - (3.0 * std::min({r, g, b}) / (r + g + b)) ;
}

double calcI(double r, double g, double b) {
    return (r + g + b) / 3.0;
}

cv::Mat rgb2hsi(cv::Mat & rgbImg) {
    cv::Mat hsiImg(rgbImg.size(), CV_32FC3);

    for (int i = 0; i < rgbImg.rows; ++i) {
        for (int j = 0; j < rgbImg.cols; ++j) {
            double b = rgbImg.at<cv::Vec3b>(i,j)[0];
            double g = rgbImg.at<cv::Vec3b>(i,j)[1];
            double r = rgbImg.at<cv::Vec3b>(i,j)[2];

            double saturation = calcS(r, g, b);
            double intensity = calcI(r, g, b);
            double hue = calcH(r, g, b);

            hsiImg.at<cv::Vec3f>(i, j) = cv::Vec3f(hue, saturation, intensity);
        }
    }

    return hsiImg;
}

int main(int argc, char ** argv) {
    if (argc == 1)
        return -1;
    cv::Mat inputImg = cv::imread(argv[1]);
    if (inputImg.empty())
        return -2;

    if (inputImg.channels() == 3)
        std::cout << "the img is colored" << std::endl;

    imshowWrapper("Input Img", inputImg);

    cv::Mat hsiImg = rgb2hsi(inputImg);
    imshowWrapper("HSI converted Img", hsiImg);

    cv::Mat hsvImg;
    cv::cvtColor(inputImg, hsvImg, cv::COLOR_BGR2HSV);

    imshowWrapper("HSV converted Img (opencv)", hsvImg);

    return 0;
}