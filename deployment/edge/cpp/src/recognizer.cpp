#include "recognizer.h"

namespace ocr {

Recognizer::Recognizer(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OCR_Recognizer") {
    // TODO: Initialize ONNX Runtime session
    // session_ = new Ort::Session(env_, model_path.c_str(), session_options_);
}

Recognizer::~Recognizer() {
    // TODO: Cleanup
    // delete session_;
}

std::string Recognizer::recognize(const cv::Mat& image) {
    // TODO: Implement recognition
    // 1. Preprocess image
    // 2. Run inference
    // 3. Decode CTC output
    return "";
}

} // namespace ocr
