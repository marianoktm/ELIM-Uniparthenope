#include <opencv2/opencv.hpp>
#include "./reusables/utils.h"

cv::Mat Harris(cv::Mat & inputImg, int sobelKernelSize, int gblurSize, float gblurSigma, float k, int thresh, bool debug = false) {
    // STEP 1: Gradient
    cv::Mat sobelGx;
    cv::Sobel(inputImg, sobelGx, CV_32F, 1, 0, sobelKernelSize, cv::BORDER_DEFAULT);
    if (debug) imshowWrapper("sobelGx", sobelGx);

    cv::Mat sobelGy;
    cv::Sobel(inputImg, sobelGy, CV_32F, 0, 1, sobelKernelSize, cv::BORDER_DEFAULT);
    if (debug) imshowWrapper("sobelGy", sobelGy);

    // STEP 2: Calculating dx^2, dy^2 and dx*dy
    cv::Mat dx2;
    cv::pow(sobelGx, 2, dx2);
    if (debug) imshowWrapper("dx2", dx2);

    cv::Mat dy2;
    cv::pow(sobelGy, 2, dy2);
    if (debug) imshowWrapper("dy2", dy2);

    cv::Mat dxdy;
    cv::multiply(sobelGx, sobelGy, dxdy);
    if (debug) imshowWrapper("dxdy", dxdy);

    // STEP 3: Applying a Gaussian filter on dx^2, dy^2 and dx*dy
    cv::Mat dx2blurred;
    cv::GaussianBlur(dx2, dx2blurred, cv::Size(gblurSize, gblurSize), gblurSigma, 0, cv::BORDER_DEFAULT);
    if (debug) imshowWrapper("dx2blurred", dx2blurred);

    cv::Mat dy2blurred;
    cv::GaussianBlur(dy2, dy2blurred, cv::Size(gblurSize, gblurSize), 0, gblurSigma, cv::BORDER_DEFAULT);
    if (debug) imshowWrapper("dy2blurred", dy2blurred);

    cv::Mat dxdyblurred;
    cv::GaussianBlur(dxdy, dxdyblurred, cv::Size(gblurSize, gblurSize), gblurSigma, gblurSigma, cv::BORDER_DEFAULT);
    if (debug) imshowWrapper("dxdyblurred", dxdyblurred);

    // STEP 5/6: Calculating and normalizing harrisResponse
    cv::Mat diag1mult;
    cv::multiply(dx2blurred, dy2blurred, diag1mult);
    cv::Mat diag2mult;
    cv::pow(dxdyblurred, 2, diag2mult);
    cv::Mat determinant;
    cv::subtract(diag1mult, diag2mult, determinant);

    cv::Mat diag1sum;
    cv::add(dx2blurred, dy2blurred, diag1sum);
    cv::Mat trace2;
    cv::pow(diag1sum, 2, trace2);

    cv::Mat harrisResponse;
    cv::subtract(determinant, k * trace2, harrisResponse);
    cv::normalize(harrisResponse, harrisResponse, 0, 255, cv::NORM_MINMAX, CV_8U, cv::Mat());
    if (debug) imshowWrapper("harrisResponse (normalized)", harrisResponse);

    // STEP 7: Applying threshold on harrisResponse and marking the angles
    cv::Mat thresholdedR;
    cv::threshold(harrisResponse, thresholdedR, thresh, 255, cv::THRESH_BINARY);
    if (debug) imshowWrapper("harrisResponse (thresholded)", thresholdedR);

    cv::Mat cornerImg = inputImg.clone();
    for (int x = 0; x < harrisResponse.rows; ++x) {
        for (int y = 0; y < harrisResponse.cols; ++y) {
            if (thresholdedR.at<uchar>(x,y) > 0) {
                cv::circle(cornerImg, cv::Point(y, x), 3, cv::Scalar(255), 1, 8, 0);
            }
        }
    }

    return cornerImg;
}


int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("inputImg", inputImg);

    int sobelKernelSize = 3;
    float gblurSigma = 2.0;
    int gblurSize = 3;
    float k = 0.05;
    int thresh = 60;

    bool isDebugMode = true;
    cv::Mat myHarrisImg = Harris(inputImg, sobelKernelSize, gblurSize, gblurSigma, k, thresh, isDebugMode);

    imshowWrapper("myHarrisImg", myHarrisImg);

    return 0;
}