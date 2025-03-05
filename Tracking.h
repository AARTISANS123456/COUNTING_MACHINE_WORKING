//#pragma once
//#ifndef TRACKING_H
//#define TRACKING_H
//
//#include <opencv2/opencv.hpp>
//#include <vector>
//#include <cmath>
//#include <map>
//
//// Data structure for a tracked part
//struct Part {
//    cv::Point2f centroid;
//    cv::Rect boundingBox;
//};
//
//// Global variables for tracking (you might later want to encapsulate these into a class)
//extern std::map<int, Part> trackedParts;
//extern int nextPartID;
//extern const int DIST_THRESHOLD;
//
//// Check if the movement is significant enough (customize sizeThreshold if needed)
//bool isSignificantMovement(const cv::Point2f& prev, const cv::Point2f& current, int sizeThreshold);
//
//// Process a frame by checking new detections and updating the tracked parts.
//// The frame is drawn with rectangles and circles on newly detected parts.
//void processFrame(cv::Mat& frame, const std::vector<cv::Rect>& detectedParts);
//
//#endif // TRACKING_H
#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <cmath>
#include <map>

// Data structure for a tracked part
struct Part {
    cv::Point2f centroid;
    cv::Rect boundingBox;
    cv::Mat descriptor; // ORB feature descriptor of the detected part
};

// Global variables for tracking (could be encapsulated later into a class)
extern std::map<int, Part> trackedParts;
extern int nextPartID;

// Threshold for deciding if two descriptors match (tune as needed)
extern const float MATCH_THRESHOLD;

// Global ORB detector/descriptor instance (initialized in the source file)
extern cv::Ptr<cv::ORB> orb;

// Function that returns true if two descriptors match well enough
bool matchDescriptor(const cv::Mat& desc1, const cv::Mat& desc2, float threshold);

// Process a frame by matching new detections with existing tracked parts using both
// spatial (centroid) and appearance (ORB descriptor) information.
// For each detected bounding box in the input vector, an ORB descriptor is computed
// (on the ROI from the input frame). Then, each new detection is compared against
// the stored descriptor of existing tracks. If a match is found (distance < threshold),
// the track is updated; otherwise, a new track is created.
// The function also draws rectangles and circles on newly created tracks.
void processFrame(cv::Mat& frame, const std::vector<cv::Rect>& detectedParts);

#endif // TRACKING_H
