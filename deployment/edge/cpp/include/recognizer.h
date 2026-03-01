#pragma once

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

namespace ocr {

class Recognizer {
public:
    Recognizer(const std::string& model_path);
    ~Recognizer();
    
    std::string recognize(const cv::Mat& image);
    
private:
    Ort::Env env_;
    Ort::Session* session_;
    Ort::SessionOptions session_options_;
    std::vector<std::string> vocabulary_;
};

} // namespace ocr
