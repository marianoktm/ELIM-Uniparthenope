#include <opencv2/opencv.hpp>
#include <vector>
#include "../reusables/utils.h"

/**
 * Applies Otsu's Thresholding algorithm to an input image.
 *
 * Otsu's Thresholding is a method for automatically determining the optimal
 * threshold value to separate foreground and background regions in a grayscale image.
 * It minimizes the intra-class variance of pixel intensities.
 *
 * This function performs the following steps:
 * 1. Computes the normalized image histogram.
 * 2. Calculates the optimal threshold value using Otsu's method.
 * 3. Applies Gaussian blur to the input image to reduce noise.
 * 4. Thresholds the blurred image using the calculated optimal threshold.
 *
 * @param input     Input image (grayscale).
 * @param blurSize  Size of the Gaussian filter kernel for smoothing (default is 3).
 * @param blurSigma Standard deviation for Gaussian blur (default is 0.5).
 *
 * @return Binary image with pixels separated into foreground and background regions.
 */
cv::Mat otsu(cv::Mat & input, int blurSize = 3, float blurSigma = 0.5) {
    cv::Mat img = input.clone();

    // Step 1: Compute the normalized image histogram.
    std::vector<double> histogram(256, 0.0);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            histogram.at(img.at<uchar>(cv::Point(x, y)))++;

    // Histogram Normalization
    int numberOfPixels = img.rows * img.cols;
    for (double & numberOfPixelsInBin : histogram)
        numberOfPixelsInBin /= numberOfPixels;

    // Step 2: Calculate the optimal threshold value using Otsu's method.
    double globalCumulativeMean = 0.0; // mG
    for (int i = 0; i < histogram.size(); ++i)
        globalCumulativeMean += i * histogram.at(i);

    // Initialize variables for probabilities, cumulative means, and maximum between-class variance.
    double probability = 0.0; // pi
    double cumulativeMean = 0.0; // m1
    double betweenClassesVariance; // sigma^2 B
    double maxVariance = 0.0; // max sigma^2 B
    int optimalTH = 0; // k*
    for (int k = 0; k < histogram.size(); ++k) {
        probability += histogram.at(k);
        cumulativeMean += k * histogram.at(k);

        // Calculate between-class variance using Otsu's method.
        double bwcVarNumerator = std::pow(globalCumulativeMean * probability - cumulativeMean, 2);
        double probability2 = 1 - probability;
        double bwcVarDenominator = probability * probability2;
        betweenClassesVariance = bwcVarNumerator / bwcVarDenominator;

        // Update the optimal threshold if a higher variance is found.
        if (betweenClassesVariance > maxVariance) {
            maxVariance = betweenClassesVariance;
            optimalTH = k;
        }
    }

    // Step 3: Apply Gaussian blur to reduce noise.
    cv::GaussianBlur(img, img, cv::Size(blurSize, blurSize), blurSigma, blurSigma);

    // Step 4: Threshold the blurred image using the calculated optimal threshold.
    cv::Mat thresholdedImg;
    cv::threshold(img, thresholdedImg, optimalTH, 255, cv::THRESH_BINARY);

    return thresholdedImg;
}

int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("Input Img", inputImg);

    cv::Mat otsuImg = otsu(inputImg);
    imshowWrapper("Otsu Img", otsuImg);
    return 0;
}

