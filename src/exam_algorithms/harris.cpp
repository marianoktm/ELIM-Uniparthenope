#include <opencv2/opencv.hpp>
#include "../reusables/utils.h"

/**
 * Applies the Harris Corner Detector algorithm to an input image.
 *
 * This function performs the following steps:
 * 1. Compute the horizontal and vertical derivatives using Sobel operators.
 * 2. Calculate the products of derivatives and their squares.
 * 3. Apply Gaussian smoothing to the derivative images.
 * 4. Compute the elements of the structure tensor.
 * 5. Compute the Harris response function R.
 * 6. Apply a threshold to detect corners.
 * 7. Draw circles on the input image at detected corner locations.
 *
 * @param input     Input image (grayscale).
 * @param k         Harris corner detector parameter (usually in the range of 0.04 to 0.06).
 * @param sobelSize Size of the Sobel kernel for derivative computation.
 * @param threshTH  Threshold for corner detection (default is 70).
 * @param blurSize  Size of the Gaussian filter kernel for smoothing (default is 3).
 * @param blurSigma Standard deviation for Gaussian blur (default is 0.5).
 *
 * @return An image with detected corners marked by circles.
 */
cv::Mat harris(cv::Mat & input, float k, int sobelSize, int threshTH = 70, int blurSize = 3, float blurSigma = 0.5) {
    cv::Mat img = input.clone();

    // Step 1: Compute horizontal and vertical derivatives.
    cv::Mat x_gradient, y_gradient;
    cv::Sobel(img, x_gradient, CV_32F, 1, 0, sobelSize);
    cv::Sobel(img, y_gradient, CV_32F, 0, 1, sobelSize);

    // Step 2: Calculate products of derivatives and their squares.
    cv::Mat gradientProduct;
    cv::multiply(x_gradient, y_gradient, gradientProduct);

    cv::pow(x_gradient, 2, x_gradient);
    cv::pow(y_gradient, 2, y_gradient);

    // Step 3: Apply Gaussian smoothing to derivative images.
    cv::GaussianBlur(x_gradient, x_gradient, cv::Size(blurSize, blurSize), blurSigma, blurSigma);
    cv::GaussianBlur(y_gradient, y_gradient, cv::Size(blurSize, blurSize), blurSigma, blurSigma);
    cv::GaussianBlur(gradientProduct, gradientProduct, cv::Size(blurSize, blurSize), blurSigma, blurSigma);

    // Step 4: Compute elements of the structure tensor.
    cv::Mat mainDiagonalProduct;
    cv::multiply(x_gradient, y_gradient, mainDiagonalProduct);
    cv::Mat secDiagonalProduct;
    cv::pow(gradientProduct, 2, secDiagonalProduct);

    // Step 5: Compute the Harris response function harrisResponse.
    cv::Mat determinant = mainDiagonalProduct - secDiagonalProduct;

    cv::Mat trace;
    cv::pow(x_gradient + y_gradient, 2, trace);

    cv::Mat harrisResponse = determinant - k * trace;
    cv::normalize(harrisResponse, harrisResponse, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Step 6: Apply a threshold to detect corners.
    cv::threshold(harrisResponse, harrisResponse, threshTH, 255, cv::THRESH_BINARY);

    // Step 7: Draw circles on the input image at detected corner locations.
    cv::Mat out = input.clone();
    for (int y = 0; y < harrisResponse.rows; ++y) {
        for (int x = 0; x < harrisResponse.cols; ++x) {
            if (harrisResponse.at<uchar>(cv::Point(x, y)) > 0) {
                cv::circle(out, cv::Point(x, y), 3, cv::Scalar(255), 1, 8, 0);
            }
        }
    }

    return out;
}

int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("Input Img", inputImg);

    float k = 0.05;
    int sobelSize = 3;
    int blurSize = 3;
    float blurSigma = 2.0;
    int threshTH = 60;

    cv::Mat harrisImg = harris(inputImg, k, sobelSize, threshTH, blurSize, blurSigma);
    imshowWrapper("Harris Img", harrisImg);

    return 0;
}