#pragma once

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

namespace ocr {

struct BoundingBox {
    float x1, y1, x2, y2;
    float confidence;
};

class Detector {
public:
    Detector(const std::string& model_path);
    ~Detector();
    
    std::vector<BoundingBox> detect(const cv::Mat& image);
    
private:
    Ort::Env env_;
    Ort::Session* session_;
    Ort::SessionOptions session_options_;
};

} // namespace ocr
