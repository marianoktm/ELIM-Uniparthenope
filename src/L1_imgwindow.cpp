#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // Argument check
    if (argc == 1) {
        return -2;
    }

    // Opening an image from command line arguments
    cv::Mat img = cv::imread(argv[1], -1);
    if (img.empty()) {
        return -1;
    }

    // Launching a window to show the img
    cv::namedWindow("EXAMPLE", cv::WINDOW_AUTOSIZE);
    // Si può mostrare un'img direttamente con imshow, senza prima fare namedwindow
    cv::imshow("EXAMPLE", img);
    cv::waitKey(0);
    cv::destroyWindow("EXAMPLE");

    // La gestione della memoria delle immagini è gestita dalla libreria.
    return 0;
}
