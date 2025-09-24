#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>

void mask_image(cv::Mat& window, cv::Mat& mask);
std::vector<cv::Point> find_horizontal_lines(cv::Mat& mask, cv::Mat& edges);
std::vector<cv::Point> RANSAC(const std::vector<cv::Point>& points,
                              int iterations, double threshold);

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Missing input video file." << std::endl;
    return -1;
  }

  const std::string source = argv[1];

  cv::VideoCapture inputVideo(source);
  if (!inputVideo.isOpened()) {
    std::cerr << "Could not open input video: " << source << std::endl;
    return -1;
  }

  int frame_width = static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height =
      static_cast<int>(inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = inputVideo.get(cv::CAP_PROP_FPS);
  int fourcc = static_cast<int>(inputVideo.get(cv::CAP_PROP_FOURCC));
  std::string output_filename = "stabilized.MOV";

  cv::VideoWriter outputVideo;
  outputVideo.open(output_filename, fourcc, fps,
                   cv::Size(frame_width, frame_height), true);
  if (!outputVideo.isOpened()) {
    std::cerr << "Could not open output video for write: " << source
              << std::endl;
    return -1;
  }

  std::string input_filename = "raw.MOV";
  cv::VideoWriter rawVideo;
  rawVideo.open(input_filename, fourcc, fps,
                cv::Size(frame_width, frame_height), true);
  if (!rawVideo.isOpened()) {
    std::cerr << "Could not open input video for write: " << source
              << std::endl;
    return -1;
  }

  cv::namedWindow("Video", cv::WINDOW_NORMAL);
  cv::resizeWindow("Video", 1280, 720);
  cv::namedWindow("Video_raw", cv::WINDOW_NORMAL);
  cv::resizeWindow("Video_raw", 1280, 720);

  cv::Mat frame, mask, edges, stablized;
  double angle = 0;

  while (true) {
    inputVideo >> frame;
    if (frame.empty()) break;

    int x = 0;
    int y = frame.rows / 2 + 350;
    int w = frame.cols;
    int h = frame.rows / 2 - 600;

    cv::Rect roi(x, y, w, h);
    cv::Mat window = frame(roi);
    // Debug: View window in frame
    // cv::rectangle(frame, cv::Point(x, y), cv::Point(w, y + h),
    //               cv::Scalar(255, 0, 0), 3);

    mask_image(window, mask);

    std::vector<cv::Point> horizontal_lines =
        find_horizontal_lines(mask, edges);

    std::vector<cv::Point> filtered = RANSAC(horizontal_lines, 100, 15);

    if (filtered.size() > 4) {
      cv::Vec4f line;
      cv::fitLine(filtered, line, cv::DIST_HUBER, 0, 0.01, 0.01);

      float vx = line[0], vy = line[1];
      float x0 = line[2], y0 = line[3];

      int y_left = cvRound(y0 - (x0 * vy / vx));
      int y_right = cvRound(y0 + ((window.cols - x0) * vy / vx));

      // Debug: show line for demo footage
      // cv::line(frame, cv::Point(0, y_left + y),
      //          cv::Point(window.cols, y_right + y), cv::Scalar(0, 255, 0), 5,
      //          cv::LINE_AA);

      int x1 = 0;
      int y1 = y_left + y;
      int x2 = window.cols;
      int y2 = y_right + y;

      double angle = (std::atan2((y2 - y1), (x2 - x1)) * 180.0) / CV_PI;

      static double smoothed_angle = 0.0;
      double alpha = 0.1;
      smoothed_angle = alpha * angle + (1.0 - alpha) * smoothed_angle;

      cv::Point2f center(frame.cols / 2.0, frame.rows / 2.0);
      cv::Mat rotation = cv::getRotationMatrix2D(center, smoothed_angle, 1.0);

      cv::warpAffine(frame, stablized, rotation, frame.size(), cv::INTER_LINEAR,
                     cv::BORDER_REFLECT);
    }

    // outputVideo << stablized;
    // rawVideo << frame;

    cv::imshow("Video_raw", frame);
    cv::imshow("Video", stablized);

    char c = (char)cv::waitKey(1);
    if (c == 'q' || c == 27) break;
  }

  inputVideo.release();
  cv::destroyAllWindows();

  return 0;
}

void mask_image(cv::Mat& window, cv::Mat& mask) {
  cv::Scalar wall_lower(90, 30, 10);
  cv::Scalar wall_upper(150, 255, 150);

  cv::inRange(window, wall_lower, wall_upper, mask);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
}

std::vector<cv::Point> find_horizontal_lines(cv::Mat& mask, cv::Mat& edges) {
  cv::Canny(mask, edges, 1, 5, 3);

  std::vector<cv::Vec4f> linesP;
  cv::HoughLinesP(edges, linesP, 1, CV_PI / 180, 40, 40, 10);

  std::vector<cv::Point> horizontal_lines;
  for (size_t i = 0; i < linesP.size(); i++) {
    cv::Vec4i l = linesP[i];

    double angle = std::atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;

    if (angle < 10 && angle > -10) {
      horizontal_lines.push_back(cv::Point(l[0], l[1]));
      horizontal_lines.push_back(cv::Point(l[2], l[3]));
    }
  }

  return horizontal_lines;
}

std::vector<cv::Point> RANSAC(const std::vector<cv::Point>& points,
                              int iterations = 100, double threshold = 15) {
  if (points.size() < 6) return points;

  std::vector<cv::Point> best_inliers;

  for (int i = 0; i < iterations; i++) {
    int idx1 = rand() % points.size();
    int idx2 = rand() % points.size();
    while (idx2 == idx1) idx2 = rand() % points.size();

    cv::Point p1 = points[idx1];
    cv::Point p2 = points[idx2];

    if (std::abs(p1.x - p2.x) < 10) continue;

    double m = (double)(p2.y - p1.y) / (p2.x - p1.x);
    double b = p1.y - m * p1.x;

    std::vector<cv::Point> inliers;
    for (const auto& p : points) {
      double expected_y = m * p.x + b;
      double distance = std::abs(p.y - expected_y);

      if (distance <= threshold) {
        inliers.push_back(p);
      }
    }

    if (inliers.size() > best_inliers.size()) {
      best_inliers = inliers;
    }
  }

  return best_inliers;
};
