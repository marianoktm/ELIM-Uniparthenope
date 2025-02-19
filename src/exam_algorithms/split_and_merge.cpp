#include <opencv2/opencv.hpp>
#include <vector>
#include "../reusables/utils.h"

#define up_left 0    //  -----
#define up_right 1   //  |0|1|
#define low_left 3   //  |3|2|
#define low_right 2  //  -----

/**
 * @class qtNode
 * @brief Represents a node in a Quadtree data structure.
 */
class qtNode {
private:
    cv::Rect region;

    qtNode * upper_left = nullptr;
    qtNode * upper_right = nullptr;
    qtNode * lower_left = nullptr;
    qtNode * lower_right = nullptr;

    std::vector<qtNode *> merged;
    std::vector<bool> isMerged = std::vector<bool>(4, false);

    double mean = -1;
    double stdDev = -1;

public:
    const cv::Rect &getRegion() const { return region; }
    void setRegion(const cv::Rect &region) { qtNode::region = region; }

    qtNode *getUpperLeft() const { return upper_left; }
    void setUpperLeft(qtNode *upperLeft) { upper_left = upperLeft; }

    qtNode *getUpperRight() const { return upper_right; }
    void setUpperRight(qtNode *upperRight) { upper_right = upperRight; }

    qtNode *getLowerLeft() const { return lower_left; }
    void setLowerLeft(qtNode *lowerLeft) { lower_left = lowerLeft; }

    qtNode *getLowerRight() const { return lower_right; }
    void setLowerRight(qtNode *lowerRight) { lower_right = lowerRight; }

    double getMean() const { return mean; }
    void setMean(double mean) { qtNode::mean = mean; }

    double getStdDev() const { return stdDev; }
    void setStdDev(double stdDev) { qtNode::stdDev = stdDev; }

    void pushOnMerged(qtNode *toPush) { merged.push_back(toPush); }
    std::vector<qtNode *> getMerged() { return merged; }

    void setIsMerged(int quadrant, bool flag) { isMerged.at(quadrant) = flag; }
    void setIsMergedAllFalse() { for (auto && i : isMerged) i = false; }
    std::vector<bool> getIsMerged() { return isMerged; }


    explicit qtNode(cv::Rect &region) { qtNode::region = region; }
};

/**
 * Check if a node std deviation is less or equal than the given threshold.
 *
 * @param node The node to check.
 * @param samTH The threshold for the predicate.
 * @return True if the node satisfies the predicate, otherwise false.
 */
bool satisfyPredicate(qtNode *node, double samTH) {
    return node->getStdDev() <= samTH;
}

/**
 * Check if a region is divisible into smaller regions.
 *
 * @param node The node representing the region.
 * @param minRegSize The minimum size of regions for divisibility.
 * @return True if the region is divisible, otherwise false.
 */
bool regionIsDivisible(qtNode *node, int minRegSize) {
    return node->getRegion().width > minRegSize;
}

/**
 * Check if a region is divisible into smaller regions.
 *
 * @param region The cv::Rect representing the region.
 * @param minRegSize The minimum size of regions for divisibility.
 * @return True if the region is divisible, otherwise false.
 */
bool regionIsDivisible(cv::Rect region, int minRegSize) {
    return region.width > minRegSize and region.height > minRegSize;
}

/**
 * Check if two regions should be divided into a bigger region.
 *
 * @param node1 The node representing the first region.
 * @param node2 The node representing the second region.
 * @param samTH The threshold for the predicate.
 * @return True if the regions should be merged, otherwise false.
 */
bool shouldBeMerged(qtNode *node1, qtNode *node2, double samTH) {
    return satisfyPredicate(node1, samTH) and satisfyPredicate(node2, samTH);
}

/**
 * Perform Quadtree splitting on an image region.
 *
 * @param img The input image.
 * @param region The region to be split.
 * @param samTH The threshold for splitting.
 * @param minRegSize The minimum region size for splitting.
 * @return The root node of the Quadtree.
 */
qtNode * split(cv::Mat &img, cv::Rect region, double samTH, int minRegSize = 2) {
    // Create a new Quadtree node representing this region
    auto node = new qtNode(region);

    // Calculate the mean and standard deviation of the image region
    cv::Mat subImg = img(region);
    cv::Scalar mean, stdDev;
    cv::meanStdDev(subImg, mean, stdDev);

    // Set the calculated mean and standard deviation for the node
    node->setMean(mean[0]);
    node->setStdDev(stdDev[0]);

    // Check if the region is divisible and the predicate is not satisfied
    if (regionIsDivisible(region, minRegSize) and not satisfyPredicate(node, samTH)) {
        // Calculate dimensions for sub-regions
        int halfWidth = region.width / 2;
        int halfHeight = region.height / 2;

        // Split the image region into four quadrants
        cv::Rect upper_left(region.x, region.y, halfWidth, halfHeight);
        node->setUpperLeft(split(img, upper_left, samTH, minRegSize));

        cv::Rect upper_right(region.x + halfWidth, region.y, halfWidth, halfHeight);
        node->setUpperRight(split(img, upper_right, samTH, minRegSize));

        cv::Rect lower_left(region.x, region.y + halfHeight, halfWidth, halfHeight);
        node->setLowerLeft(split(img, lower_left, samTH, minRegSize));

        cv::Rect lower_right(region.x + halfWidth, region.y + halfHeight, halfWidth, halfHeight);
        node->setLowerRight(split(img, lower_right, samTH, minRegSize));
    }

    return node;
}

/**
 * Merge regions in a Quadtree based on a given threshold.
 *
 * @param node The current node to merge.
 * @param samTH The threshold for merging.
 * @param minRegSize The minimum region size for merging.
 */
void merge(qtNode *node, double samTH, int minRegSize) {
    // If the region is not divisible or the predicate is satisfied
    // do not merge the subregions of the node
    if (not regionIsDivisible(node, minRegSize) or satisfyPredicate(node, samTH)) {
        node->pushOnMerged(node);
        node->setIsMergedAllFalse();
        return;
    }

    // Retrieve pointers to the four child nodes
    auto upper_left = node->getUpperLeft();
    auto upper_right = node->getUpperRight();
    auto lower_left = node->getLowerLeft();
    auto lower_right = node->getLowerRight();

    // Check for merging along the UPPER LINE
    if (shouldBeMerged(upper_left, upper_right, samTH)) {
        // Merge the upper-left and upper-right child nodes
        node->pushOnMerged(upper_left);
        node->pushOnMerged(upper_right);
        node->setIsMerged(up_left, true);
        node->setIsMerged(up_right, true);

        // Also check if lower line should be merged
        if (shouldBeMerged(lower_left, lower_right, samTH)) {
            node->pushOnMerged(lower_right);
            node->pushOnMerged(lower_left);
            node->setIsMerged(low_right, true);
            node->setIsMerged(low_left, true);
        }
        else {
            // Recursively merge the lower-left and lower-right child nodes
            merge(lower_left, samTH, minRegSize);
            merge(lower_right, samTH, minRegSize);
        }
    }
    // Check for merging along the RIGHT LINE
    else if (shouldBeMerged(upper_right, lower_right, samTH)) {
        // Merge the upper-right and lower-right child nodes
        node->pushOnMerged(upper_right);
        node->pushOnMerged(lower_right);
        node->setIsMerged(up_right, true);
        node->setIsMerged(low_right, true);

        // Also check if left line should be merged
        if (shouldBeMerged(upper_left, lower_left, samTH)) {
            node->pushOnMerged(upper_left);
            node->pushOnMerged(lower_left);
            node->setIsMerged(up_left, true);
            node->setIsMerged(low_left, true);
        }
        else {
            // Recursively merge the upper-left and lower-left child nodes
            merge(upper_left, samTH, minRegSize);
            merge(lower_left, samTH, minRegSize);
        }
    }
    // Check for merging along the LOW LINE
    else if (shouldBeMerged(upper_right, lower_right, samTH)) {
        node->pushOnMerged(lower_left);
        node->pushOnMerged(lower_right);
        node->setIsMerged(low_left, true);
        node->setIsMerged(low_right, true);
        // Also check if upper line should be merged
        if (shouldBeMerged(upper_left, upper_right, samTH)) {
            node->pushOnMerged(upper_left);
            node->pushOnMerged(upper_right);
            node->setIsMerged(up_left, true);
            node->setIsMerged(up_right, true);
        }
        else {
            // Recursively merge the upper-left and upper-right child nodes
            merge(upper_left, samTH, minRegSize);
            merge(upper_right, samTH, minRegSize);
        }
    }
    // Check for merging along the LEFT LINE
    else if (shouldBeMerged(upper_left, lower_left, samTH)) {
        node->pushOnMerged(upper_left);
        node->pushOnMerged(lower_left);
        node->setIsMerged(up_left, true);
        node->setIsMerged(low_left, true);
        // Also check if right line should be merged
        if (shouldBeMerged(upper_right, lower_right, samTH)) {
            node->pushOnMerged(upper_right);
            node->pushOnMerged(lower_right);
            node->setIsMerged(up_right, true);
            node->setIsMerged(low_right, true);
        }
        else {
            // Recursively merge the upper-right and lower-right child nodes
            merge(upper_right, samTH, minRegSize);
            merge(lower_right, samTH, minRegSize);
        }
    }
    // If none of the above conditions are met, apply the merge procedure on all children individually
    else {
        merge(upper_left, samTH, minRegSize);
        merge(upper_right, samTH, minRegSize);
        merge(lower_right, samTH, minRegSize);
        merge(lower_left, samTH, minRegSize);
    }
}

/**
 * Draw merged regions onto an image.
 *
 * @param img The image to draw on.
 * @param node The Quadtree node.
 */
void draw(cv::Mat &img, qtNode * node) {
    // If the node is nullptr, return and stop processing
    if (node == nullptr)
        return;

    // Get the list of merged subregions
    auto mergedVector = node->getMerged();

    // If the mergedVector is empty, recursively call draw procedure to it's children
    if (mergedVector.empty()) {
        auto upper_left = node->getUpperLeft();
        auto upper_right = node->getUpperRight();
        auto lower_left = node->getLowerLeft();
        auto lower_right = node->getLowerRight();

        draw(img, upper_left);
        draw(img, upper_right);
        draw(img, lower_left);
        draw(img, lower_right);

        return;
    }

    // Calculate the region value based on the mean of merged nodes
    double regionValue = 0.0;
    for (auto mergedNode : mergedVector)
        regionValue += mergedNode->getMean();
    regionValue /= (int) mergedVector.size();

    // Set the region of merged nodes to the calculated region value
    for (auto mergedNode : mergedVector)
        img(mergedNode->getRegion()) = (int) regionValue;

    // If no merging occurred, return
    if (mergedVector.size() <= 1)
        return;

    // Check if each quadrant (up_left, up_right, low_right, low_left) should be drawn
    if (not node->getIsMerged().at(up_left)) {
        auto upper_left = node->getUpperLeft();
        draw(img, upper_left);
    }
    if (not node->getIsMerged().at(up_right)) {
        auto upper_right = node->getUpperRight();
        draw(img, upper_right);
    }
    if (not node->getIsMerged().at(low_right)) {
        auto lower_right = node->getLowerRight();
        draw(img, lower_right);
    }
    if (not node->getIsMerged().at(low_left)) {
        auto lower_left = node->getUpperLeft();
        draw(img, lower_left);
    }
}

/**
 * Perform Split and Merge image segmentation.
 *
 * @param inputImg The input image.
 * @param samTH The threshold for splitting and merging.
 * @param minRegSize The minimum region size for splitting.
 * @return The segmented image.
 */
cv::Mat split_and_merge(cv::Mat & inputImg, double samTH, int minRegSize = 2) {
    cv::Mat img = inputImg.clone();

    // Calculating the scaling factor for x and y axis
    // and resizing the image to a square for split_and_merge
    int squareSize = std::max(img.rows, img.cols);
    double xScaling = (double) squareSize / img.cols;
    double yScaling = (double) squareSize / img.rows;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), xScaling, yScaling);

    // Applying split and merge algorithm and drawing
    // segmented regions on the image
    cv::Rect startingRegion = cv::Rect(0, 0, resized.cols, resized.rows);

    auto quadTreeRoot = split(resized, startingRegion, samTH, minRegSize);
    merge(quadTreeRoot, samTH, minRegSize);
    draw(resized, quadTreeRoot);

    // Calculating the inverse scaling factor for x and y axis
    // and resizing the segmented image to it's original size
    double xScalingInverted = (double) img.cols / squareSize;
    double yScalingInverted = (double) img.rows / squareSize;

    cv::Mat out;
    cv::resize(resized, out, cv::Size(), xScalingInverted, yScalingInverted);

    return out;
}

int main(int argc, char **argv) {
    cv::Mat inputImg = imreadWrapper(argc, argv, cv::IMREAD_GRAYSCALE);
    imshowWrapper("Input Img", inputImg);

    int blurSize = 3;
    double blurSigma = 1.0;
    cv::GaussianBlur(inputImg, inputImg, cv::Size(blurSize, blurSize), blurSigma, blurSigma);

    double samTH = 20.0;
    int minRegSize = 4;

    cv::Mat samImg = split_and_merge(inputImg, samTH, minRegSize);
    imshowWrapper("Split and Merge Img", samImg);

    return 0;
}