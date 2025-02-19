#include <opencv2/opencv.hpp>
#include <queue>
#include "../reusables/utils.h"

/**
 * Checks if a given point is within the bounds of an image.
 *
 * @param img   Input image.
 * @param neigh Point to be checked.
 *
 * @return True if the point is within the image bounds, false otherwise.
 */
bool inRange(cv::Mat & img, cv::Point neigh) {
    return neigh.x >= 0 and neigh.x < img.cols and neigh.y >= 0 and neigh.y < img.rows;
}

/**
 * Checks if the intensity of a given pixel is similar to that of the seed pixel.
 *
 * @param img     Input image.
 * @param seed    Seed point.
 * @param neigh   Neighbor point to be checked.
 * @param similTH Intensity similarity threshold.
 *
 * @return True if the intensity difference is within the threshold, false otherwise.
 */
bool isSimilar(cv::Mat & img, cv::Point seed, cv::Point neigh, int similTH) {
    int seedIntensity = img.at<uchar>(seed);
    int currIntensity = img.at<uchar>(neigh);
    int intensityDelta = std::abs(seedIntensity - currIntensity);
    return intensityDelta < similTH;
}

/**
 * Performs region growing on a grayscale image starting from a seed point.
 *
 * Region growing is a region-based image segmentation technique that groups
 * neighboring pixels with similar intensity values into regions.
 *
 * @param input    Input grayscale image.
 * @param similTH  Intensity similarity threshold for region growing.
 * @param seed     Seed point (default is the top-left corner, cv::Point(0, 0)).
 *
 * @return Binary image highlighting the segmented region.
 */
cv::Mat region_growing(cv::Mat& input, int similTH, cv::Point seed = cv::Point(0, 0)) {
    cv::Mat img = input.clone();

    // Create an output image with the same size, initialized as black.
    cv::Mat segmentedImg = cv::Mat::zeros(img.size(), CV_8U);

    // Initialize a queue for pixel traversal, starting from the seed point.
    std::queue<cv::Point> pixelQueue;
    pixelQueue.push(seed);

    // Perform region growing using a breadth-first search (BFS) approach.
    while (not pixelQueue.empty()) {
        cv::Point currentPixel = pixelQueue.front();
        pixelQueue.pop();

        // Check if the current pixel has not been visited.
        if (segmentedImg.at<uchar>(currentPixel) == 0) {
            segmentedImg.at<uchar>(currentPixel) = 255; // Mark the pixel as part of the segmented region.

            // Define a 3x3 region of interest around the current pixel.
            cv::Rect regionOfInterest(currentPixel.x - 1, currentPixel.y - 1, 3, 3);

            // Iterate through the neighboring pixels within the region of interest.
            for (int roi_y = regionOfInterest.y; roi_y < regionOfInterest.y + regionOfInterest.height; ++roi_y) {
                for (int roi_x = regionOfInterest.x; roi_x < regionOfInterest.x + regionOfInterest.width; ++roi_x) {
                    cv::Point neighborPixel(roi_x, roi_y);

                    // Check if the neighboring pixel is within the image bounds
                    // and has similar intensity to the seed pixel.
                    if (inRange(img, neighborPixel) and isSimilar(img, seed, neighborPixel, similTH)) {
                        // Add the neighboring pixel to the queue.
                        pixelQueue.push(neighborPixel);
                    }
                }
            }
        }
    }

    return segmentedImg;
}

int main(int argc, char ** argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("Input Img", inputImg);

    int seedx = 20;
    int seedy = 40;
    cv::Point seed(seedx, seedy);
    int similTH = 60;

    cv::Mat regGrowImg = region_growing(inputImg, similTH, seed);
    imshowWrapper("Region Growing Img", regGrowImg);

    return 0;
}
