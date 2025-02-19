#include <opencv2/opencv.hpp>
#include "./reusables/utils.h"

void gradientEdgeFindingMain(cv::Mat & inputImg, int filterSize, int thresh) {
    imshowWrapper("inputImg", inputImg);

    // Applying gaussian blur
    cv::Mat gblurImg;
    cv::GaussianBlur(inputImg, gblurImg, cv::Size(filterSize, filterSize), 0, 0);
    imshowWrapper("gblurImg", gblurImg);

    // Applying Sobel
    cv::Mat sobelDx;
    cv::Sobel(gblurImg, sobelDx, CV_32FC1, 0, 1);
    imshowWrapper("sobelDx", sobelDx);

    cv::Mat sobelDy;
    cv::Sobel(gblurImg, sobelDy, CV_32FC1, 1, 0);
    imshowWrapper("sobelDy", sobelDy);

    // Sobel magnitude
    cv::Mat sobelMagnitude = cv::abs(sobelDx) + cv::abs(sobelDy);
    cv::normalize(sobelMagnitude, sobelMagnitude, 0, 255, cv::NORM_MINMAX, CV_8U);
    imshowWrapper("sobelMagnitude", sobelMagnitude);

    // Gradient edge finding
    cv::Mat orientations;
    cv::phase(sobelDx, sobelDy, orientations, true);
    cv::normalize(orientations, orientations, 0, 255, cv::NORM_MINMAX, CV_8U);
    imshowWrapper("orientations", orientations);

    // Magnitude thresholding
    cv::Mat magnitudeThresh;
    cv::threshold(sobelMagnitude, magnitudeThresh, thresh, 255, cv::THRESH_BINARY);
    imshowWrapper("magnitudeThresh", magnitudeThresh);
}

cv::Mat zeroCrossing(cv::Mat inputImg) {
    cv::Mat outImg = cv::Mat::zeros(inputImg.rows, inputImg.cols, CV_8U);

    double min;
    double max;
    cv::minMaxLoc(inputImg, &min, &max);

    float thresh = max * 0.05;

    for (int x = 1; x < inputImg.rows - 1; ++x) {
        for (int y = 1; y < inputImg.cols - 1; ++y) {
            cv::Mat neighborhood = inputImg(cv::Rect(y-1, x-1, 2, 2));
            cv::minMaxLoc(neighborhood, &min, &max);

            bool flag;
            if (inputImg.at<float>(x, y) > 0)
                flag = (min < 0);
            else
                flag = (max > 0);

            if (max - min > thresh and flag)
                outImg.at<uchar>(x, y) = 255;
        }
    }
    return outImg;
}

void laplacianOfGaussianMain(cv::Mat & inputImg, int filterSize, int sigma) {
    imshowWrapper("inputImg", inputImg);

    cv::Mat gaussianImg;
    cv::filter2D(inputImg, gaussianImg, CV_32F, cv::getGaussianKernel(filterSize, sigma));
    imshowWrapper("gaussianImg", gaussianImg);

    cv::Mat laplacianImg;
    cv::Laplacian(gaussianImg, laplacianImg, CV_32FC1, filterSize);
    imshowWrapper("laplacianImg", laplacianImg);

    cv::Mat zeroCrossingImg = zeroCrossing(laplacianImg);
    imshowWrapper("zeroCrossingImg", zeroCrossingImg);
}

int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);

    // Gradient edge finding
    int gblurFilterSize = 65;
    int tresh = 30;
    gradientEdgeFindingMain(inputImg, gblurFilterSize, tresh);

    // Laplacian of Gaussian
    int sigma = 3;
    int gkernelFilterSize = 6 * sigma;
    if (gkernelFilterSize % 2 == 0)
        gkernelFilterSize++;
    laplacianOfGaussianMain(inputImg, gkernelFilterSize, sigma);

    return 0;
}