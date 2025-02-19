#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "../reusables/utils.h"

/**
 * Applies the numberOfClusters-means clustering algorithm to a grayscale image.
 *
 * @param input             The input grayscale image to be clustered.
 * @param numberOfClusters  The number of clusters to create.
 * @param maxIterations     The maximum number of iterations for the algorithm.
 * @param deltaTH           The threshold for center updates to stop iterations.
 * @return                  The clustered grayscale image.
 */
cv::Mat kmeans_gray(cv::Mat &input, int numberOfClusters, int maxIterations, double deltaTH = 1.0) {
    cv::Mat img = input.clone();

    // Step 1: Initialize cluster centres randomly.
    srand(time(nullptr) + 1);
    std::vector<uchar> centres(numberOfClusters);
    for (int i = 0; i < numberOfClusters; ++i) {
        int x = rand() % img.cols;
        int y = rand() % img.rows;
        centres.at(i) = img.at<uchar>(cv::Point(x, y));
    }

    // Iterate until cluster centres stabilize or until maxIterations is reached
    int iterations = 0;
    int closestIndex = 0;
    bool isCentreUpdated = true;

    std::vector<std::vector<cv::Point>> clusters(numberOfClusters);
    cv::Mat clusteredImg = img.clone();

    while (isCentreUpdated or iterations < maxIterations) {
        // Resetting centre update flag and cluster containers
        isCentreUpdated = false;

        for (int i = 0; i < numberOfClusters; ++i)
            clusters.at(i).clear();

        // Step 2: Assign each pixel to the closest cluster centre
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                int currentDistance;
                int minDistance = INFINITY;
                for (int i = 0; i < numberOfClusters; ++i) {
                    currentDistance = std::abs(centres.at(i) - img.at<uchar>(cv::Point(x, y)));
                    if (currentDistance < minDistance) {
                        minDistance = currentDistance;
                        closestIndex = i;
                    }
                }
                clusters.at(closestIndex).emplace_back(x, y);
                clusteredImg.at<uchar>(cv::Point(x, y)) = centres.at(closestIndex);
            }
        }

        // Step 3: Check if centres must be updated. Calculate the new mean of each cluster.
        for (int i = 0; i < numberOfClusters; ++i) {
            if (not clusters.at(i).empty()) {
                int intensitySum = 0;
                for (auto & point : clusters.at(i))
                    intensitySum += img.at<uchar>(point);

                // Using the sum to calculate the new mean and the delta between the old and new mean
                double currentMean = intensitySum / clusters.at(i).size();
                int delta = cvRound(std::abs(currentMean - centres.at(i)));

                if (delta > deltaTH) {
                    centres.at(i) = cvRound(currentMean);
                    isCentreUpdated = true;
                }
            }
        }
        iterations++;
    }

    return clusteredImg;
}

int main(int argc, char **argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("Input Img", inputImg);

    int k = 3;
    int maxIterations = 30;
    double deltaTH = 1.0;

    cv::Mat kmeansImg = kmeans_gray(inputImg, k, maxIterations, deltaTH);
    imshowWrapper("K-Means (grayscale) Img", kmeansImg);
}