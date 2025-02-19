#include <opencv2/opencv.hpp>
#include <vector>
#include "../reusables/utils.h"

/**
 * Applies Otsu's Two-Threshold Thresholding algorithm (Otsu2k) to an input image.
 *
 * Otsu's Two-Threshold Thresholding is an extension of Otsu's method that divides
 * the image into three regions: background, middleground, and foreground.
 * It finds two threshold values to separate these regions optimally.
 *
 * This function performs the following steps:
 * 1. Computes the normalized image histogram.
 * 2. Calculates the global cumulative mean.
 * 3. Iteratively computes three probabilities and cumulative means for different
 *    threshold combinations, seeking to maximize between-class variance.
 * 4. Applies Gaussian blur to the input image to reduce noise.
 * 5. Thresholds the blurred image using the two optimal threshold values.
 *
 * @param input     Input image (grayscale).
 * @param blurSize  Size of the Gaussian filter kernel for smoothing (default is 3).
 * @param blurSigma Standard deviation for Gaussian blur (default is 0.5).
 *
 * @return Binary image with pixels separated into background, object, and foreground regions.
 */
cv::Mat otsu2k(cv::Mat& input, int blurSize = 3, float blurSigma = 0.5) {
    cv::Mat img = input.clone(); // Clone the input image to prevent modification.

    // Step 1: Compute the normalized image histogram.
    std::vector<double> histogram(256, 0.0);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            histogram.at(img.at<uchar>(cv::Point(x, y)))++; // Increment the histogram bin for each pixel value.

    // Histogram Normalization
    int numberOfPixels = img.rows * img.cols;
    for (double & bin : histogram)
        bin /= numberOfPixels;

    // Step 2: Calculate the global cumulative mean.
    double globalCumulativeMean = 0.0;
    for (int i = 0; i < histogram.size(); ++i)
        globalCumulativeMean += i * histogram.at(i);

    // Initialize variables for probabilities, cumulative means, and maximum between-class variance.
    std::vector<double> probabilities(3, 0.0);
    std::vector<double> cumulativeMeans(3, 0.0);
    double betweenClassesVariance;
    double maxVariance = 0.0;
    std::vector<int> optimalTH(2, 0);

    // Step 3: Iteratively compute optimal threshold values to maximize between-class variance.
    for (int k1 = 0; k1 < histogram.size() - 2; ++k1) {
        probabilities.at(0) += histogram.at(k1);
        cumulativeMeans.at(0) += k1 * histogram.at(k1);

        for (int k2 = k1 + 1; k2 < histogram.size() - 1; ++k2) {
            probabilities.at(1) += histogram.at(k2);
            cumulativeMeans.at(1) += k2 * histogram.at(k2);

            for (int k = k2 + 1; k < histogram.size(); ++k) {
                probabilities.at(2) += histogram.at(k);
                cumulativeMeans.at(2) += k * histogram.at(k);

                // Calculate between-class variance using Otsu's method.
                betweenClassesVariance = 0.0;
                for (int i = 0; i < 3; ++i) {
                    // Pi * (mi - mg)^2
                    double currentCumulativeMean = cumulativeMeans.at(i) / probabilities.at(i); // mi = 1/P1 * m
                    betweenClassesVariance += probabilities.at(i) * std::pow(currentCumulativeMean - globalCumulativeMean, 2);
                }

                // Update the optimal threshold values if higher variance is found.
                if (betweenClassesVariance > maxVariance) {
                    maxVariance = betweenClassesVariance;
                    optimalTH.at(0) = k1;
                    optimalTH.at(1) = k2;
                }
            }
            probabilities.at(2) = 0.0;
            cumulativeMeans.at(2) = 0.0;
        }
        probabilities.at(1) = 0.0;
        cumulativeMeans.at(1) = 0.0;
    }

    // Step 4: Apply Gaussian blur to reduce noise.
    cv::GaussianBlur(img, img, cv::Size(blurSize, blurSize), blurSigma, blurSigma);

    // Step 5: Threshold the blurred image using the two optimal threshold values.
    cv::Mat thresholdedImg = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    for (int x = 0; x < img.rows; ++x) {
        for (int y = 0; y < img.cols; ++y) {
            if (img.at<uchar>(x, y) >= optimalTH.at(1))
                thresholdedImg.at<uchar>(x, y) = 255; // Foreground pixel.
            else if (img.at<uchar>(x, y) >= optimalTH.at(0))
                thresholdedImg.at<uchar>(x, y) = (255 + 1) / 2; // Middleground pixel.
            // Background pixels remain black (0).
        }
    }

    return thresholdedImg;
}

int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("Input Img", inputImg);

    cv::Mat otsu2kImg = otsu2k(inputImg);
    imshowWrapper("Otsu2K Img", otsu2kImg);
    return 0;
}