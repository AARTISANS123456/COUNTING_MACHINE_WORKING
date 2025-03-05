//#include "Tracking.h"
//
//std::map<int, Part> trackedParts;
//int nextPartID = 0;
//const int DIST_THRESHOLD = 200;
//
//bool isSignificantMovement(const cv::Point2f& prev, const cv::Point2f& current, int sizeThreshold) {
//    return cv::norm(prev - current) > sizeThreshold;
//}
//
//void processFrame(cv::Mat& frame, const std::vector<cv::Rect>& detectedParts) {
//    std::vector<Part> newParts;
//    // Loop over all detected bounding boxes
//    for (const auto& bbox : detectedParts) {
//        cv::Point2f centroid(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
//        bool isNew = true;
//        // Iterate over currently tracked parts
//        for (auto it = trackedParts.begin(); it != trackedParts.end(); /* no increment here */) {
//            // If the movement is significant, remove the old tracked part
//            if (isSignificantMovement(it->second.centroid, centroid, DIST_THRESHOLD)) {
//                it = trackedParts.erase(it);
//            }
//            else {
//                isNew = false;
//                ++it;
//            }
//        }
//        // If the detection is new, add it to the tracked parts
//        if (isNew) {
//            trackedParts[nextPartID++] = { centroid, bbox };
//            newParts.push_back({ centroid, bbox });
//        }
//    }
//
//    // Draw the new parts on the frame
//    for (const auto& part : newParts) {
//        cv::rectangle(frame, part.boundingBox, cv::Scalar(0, 255, 0), 2);
//        cv::circle(frame, part.centroid, 5, cv::Scalar(0, 0, 255), -1);
//    }
//}
#include "Tracking.h"

// Initialize globals
std::map<int, Part> trackedParts;
int nextPartID = 0;
const float MATCH_THRESHOLD = 50.0f; // Adjust this threshold as needed

// Create a global ORB detector/descriptor instance
cv::Ptr<cv::ORB> orb = cv::ORB::create();

// matchDescriptor: compares two ORB descriptors using BFMatcher with Hamming norm.
// Returns true if the best match distance is less than the given threshold.
bool matchDescriptor(const cv::Mat& desc1, const cv::Mat& desc2, float threshold) {
    // Check if either descriptor is empty.
    if (desc1.empty() || desc2.empty())
        return false;

    // Use BFMatcher with cross-check enabled for robust matching.
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(desc1, desc2, matches);

    if (matches.empty())
        return false;

    // Find the best (minimum) distance among matches.
    float bestDist = std::numeric_limits<float>::max();
    for (const auto& m : matches) {
        if (m.distance < bestDist)
            bestDist = m.distance;
    }
    return bestDist < threshold;
}

void processFrame(cv::Mat& frame, const std::vector<cv::Rect>& detectedParts) {
    // We'll collect new tracks in this vector.
    std::vector<Part> newParts;

    // Iterate through all detected bounding boxes.
    for (const auto& bbox : detectedParts) {
        // Compute the centroid.
        cv::Point2f centroid(bbox.x + bbox.width / 2.0f, bbox.y + bbox.height / 2.0f);

        // Extract the region of interest (ROI) from the frame.
        cv::Mat roi = frame(bbox).clone();
        cv::Mat gray;
        // Convert ROI to grayscale if needed.
        if (roi.channels() == 3)
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        else
            gray = roi;

        // Detect keypoints and compute descriptors using ORB.
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        orb->detectAndCompute(gray, cv::noArray(), keypoints, descriptor);

        // Skip this detection if no descriptor could be computed.
        if (descriptor.empty())
            continue;

        bool matched = false;
        // Try to match the new detection against existing tracks.
        for (auto& kv : trackedParts) {
            // Use our matching function.
            if (matchDescriptor(kv.second.descriptor, descriptor, MATCH_THRESHOLD)) {
                // If a match is found, update the tracked part's centroid, bounding box, and descriptor.
                kv.second.centroid = centroid;
                kv.second.boundingBox = bbox;
                kv.second.descriptor = descriptor.clone();
                matched = true;
                break;
            }
        }

        // If no match was found, consider this detection a new part.
        if (!matched) {
            Part newPart;
            newPart.centroid = centroid;
            newPart.boundingBox = bbox;
            newPart.descriptor = descriptor.clone();
            trackedParts[nextPartID++] = newPart;
            newParts.push_back(newPart);
        }
    }

    // Optionally, draw only the newly added parts (or you could iterate over all tracks).
    for (const auto& part : newParts) {
        cv::rectangle(frame, part.boundingBox, cv::Scalar(0, 255, 0), 2);
        cv::circle(frame, part.centroid, 5, cv::Scalar(0, 0, 255), -1);
    }
}
