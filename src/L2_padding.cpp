#include <opencv2/opencv.hpp>

#define B 0
#define R 1
#define G 2

cv::Mat zeroPaddingCustom(const cv::Mat & inputImg,
                            int top,
                            int bottom,
                            int left,
                            int right) {
    int newHeight = inputImg.rows + top + bottom;
    int newWidth = inputImg.cols + left + right;

    cv::Mat paddedImg(newHeight, newWidth, inputImg.type());
    // cv::Rect(x, y, width, height) usata per operare su una regione di un'img. Qui copiamo inputImg in (x,y) di paddedImg
    inputImg.copyTo(paddedImg(cv::Rect(left, top, inputImg.cols, inputImg.rows)));

    // cv::rectangle(img, rec, color, thickness) disegna un rettangolo colorato con color di coordinate rec sull'img data in input. thickness -1 significa che il rettangolo viene riempito.
    // bordo superiore
    cv::rectangle(paddedImg,
                  cv::Rect(0, 0, newWidth, top),
                  cv::Scalar(0, 0, 0),
                  -1);

    // bordo inferiore
    cv::rectangle(paddedImg,
                  cv::Rect(newHeight-bottom,0, newWidth, bottom),
                  cv::Scalar(0, 0, 0),
                  -1);

    // bordo sinistro
    cv::rectangle(paddedImg,
                  cv::Rect(0, top, left, inputImg.rows),
                  cv::Scalar(0, 0, 0),
                  -1);

    // bordo destro
    cv::rectangle(paddedImg,
                  cv::Rect(inputImg.rows + left, top, right, inputImg.rows),
                  cv::Scalar(0, 0, 0),
                  -1);

    return paddedImg;
}

cv::Mat zeroPaddingCustom(const cv::Mat & inputImg, int padding) {
    return zeroPaddingCustom(inputImg, padding, padding, padding, padding);
}

cv::Mat meanxbyx(const cv::Mat & inputImg, int maskSize) {
    cv::Mat averagedImg(inputImg.rows, inputImg.cols, inputImg.type());

    int padding = maskSize - 1 / 2;
    cv::Mat paddedImg = zeroPaddingCustom(inputImg, padding);

    for (int row = 0; row < inputImg.rows; ++row) {
        for (int col = 0; col < inputImg.cols; ++col) {
            cv::Mat regionOfInterest = paddedImg(cv::Rect(col, row, maskSize, maskSize));
            cv::Scalar meanColor = cv::mean(regionOfInterest);
            averagedImg.at<cv::Vec3b>(row, col) = cv::Vec3b(meanColor[B], meanColor[R], meanColor[G]);
        }
    }

    return averagedImg;
}

int main(int argc, char** argv) {
    if (argc == 1) return -2;

    // Input img
    cv::Mat inputImg = cv::imread(argv[1], -1);
    if (inputImg.empty()) return -1;

    cv::imshow("inputImg", inputImg);
    cv::waitKey(0);
    cv::destroyWindow("inputImg");

    // Adding padding to the img
    cv::Mat paddedImg = zeroPaddingCustom(inputImg, 15, 20, 40, 80);

    cv::imshow("paddedImg", paddedImg);
    cv::waitKey(0);
    cv::destroyWindow("paddedImg");

    // 15x15 mean value
    cv::Mat averagedImg = meanxbyx(inputImg, 25);

    cv::imshow("averagedImg", averagedImg);
    cv::waitKey(0);
    cv::destroyWindow("averagedImg");

    return 0;
}
