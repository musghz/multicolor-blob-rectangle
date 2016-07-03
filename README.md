# multicolor-blob-rectangle
Trackng a multicolor marker by manipulating the detected bounding boxes for each color channel.


/// @file colorCodeTrackingAlgoV02.cpp
///
/// @brief Blob tracking using 2-color-codes.
///
/// Adapted from blobTrackingV04.cpp. Inspired by Khan (2015) but
/// different in the sense that instead of dilating and performing
/// logical operations on the entire thresholded image, this is
/// done on bounding boxes represented by rect objects.
///
/// Starts in tracking mode. Right click in the color window
/// to toggle between calibration mode and tracking mode. Tracking
/// color channel thresholds initialized from babyMotionConfig.txt
/// and when the program quits, the thresholds are written to the
/// file, overwriting existing values.
///
/// Calibration mode:
/// Press 1, 2, or 3 key for channel 1, 2, or 3 to select calibration
/// channel. Then right click color window to enter calibration mode.
/// To select color for a channel, drag the cursor over area with
/// desired color. Right click again to exit calibration mode. For
/// next channel, press the desired colored channel key and repeat
/// the process. While dragging the reactangle to select color, the
/// terminal will display some stats. The color window is live feed
/// so do not move the colored object or the camera. The bounding box
/// for the calibration channel will show up as a rectangle covering
/// that area.
///
/// Tracking mode:
/// Once the bounding box adequately covers the desired color,
/// right-click to enter tracking mode. The thresholded image
/// window will demonstrate thresholding according to the max
/// and min HSV values obtained from the bounding box. Press a
/// channel number key for a different channel and  then right
/// click to enter calibration mode again.
///
/// References:
/// http://docs.opencv.org/3.1.0/d7/d1d/tutorial_hull.html#gsc.tab=0
/// http://docs.opencv.org/3.1.0/da/d0c/tutorial_bounding_rects_circles.html#gsc.tab=0
/// http://docs.opencv.org/3.1.0/d0/d49/tutorial_moments.html#gsc.tab=0
///
/// Created 27 May 2016 (V02)
