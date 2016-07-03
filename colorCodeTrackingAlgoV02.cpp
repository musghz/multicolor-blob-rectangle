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
///
/// @author Mustafa Ghazi

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace cv;
using namespace std;

// throw away blobs smaller than this
// 64 for 640x480
// smaller for 320x240
#define MINAREABLOB 64

int mouseDraggedFlag = 0; // detects mouse dragged event
int trackModeFlag = 1; // keeps track of whether calibrating (0) or tracking (1)
int channelFlag = 0; // keeps track of current channel being calibrated
char charCheckForKey = 0;
int BBOX[4] = {0,0,1,1}; // bounding box for calibration x1,y1,x2,y2
int frameCount = 0; // keeps track of frame count
Mat imgOriginal;		// input image
Mat imgHSV;
Mat imgThresh;
Mat imgThreshCh1;
Mat imgThreshCh2;
Mat imgThreshCh3;

static void onMouse(int event, int x, int y, int f, void*);
int getChannelFlag(char charKey);
void getBoundingBoxHSV(Mat myImgHSV, int BOX[], int HSVMIN[], int HSVMAX[]);
void getThresholdRects(Mat myImgThresh, vector<Rect> &filteredRect);
void detectCCBlobs(int MINHSV[][3], int MAXHSV[][3]);
void dilateRects(int factor, vector<Rect> &myRect);
int getCCRectBinary(vector<Rect> &rectsChA, vector<Rect> &rectsChB, vector<int> &usedA, vector<int> &usedB, Rect ccRects[], int code);
void detectBlobs(int MINHSV[], int MAXHSV[]);
int loadConfigFile(int MIN[][3], int MAX[][3], int nChannels);
int saveConfigFile(int MIN[][3], int MAX[][3], int nChannels);

int main(int argc, char* argv[]) {
	double ticks = (double)getTickCount();
	cv::VideoCapture capWebcam(0);		// declare a VideoCapture object and associate to webcam, 0 => use 1st webcam
	if (capWebcam.isOpened() == false) {				// check if VideoCapture object was associated to webcam successfully
		std::cout << "error: capWebcam not accessed successfully\n\n";	// if not, print error message to std out
		return(0);														// and exit program
	}

	int HSVMAX[3] = { 0, 0, 0 };
	int HSVMIN[3] = { 255,255,255 };
	int HSVMAXALL[3][3] = {{0,0,0}, {0,0,0}, {0,0,0}}; // HSV for 3 channels // HSV max thresh for all channesl
	int HSVMINALL[3][3] = {{255, 255, 255}, {255, 255, 255}, {255, 255, 255}}; // HSV min thresh for 3 channels
	loadConfigFile(HSVMINALL,HSVMAXALL,3); // save HSV thresholds to config file

	// declare windows
	namedWindow("imgOriginal", CV_WINDOW_AUTOSIZE);	// note: you can use CV_WINDOW_NORMAL which allows resizing the window
	//namedWindow("imgThresh", CV_WINDOW_AUTOSIZE);	// or CV_WINDOW_AUTOSIZE for a fixed size window matching the resolution of the image
													// CV_WINDOW_AUTOSIZE is the default
	//namedWindow("imgHSV", CV_WINDOW_AUTOSIZE);
	//set the callback function for any mouse event
	setMouseCallback("imgOriginal", onMouse, NULL);

	while (charCheckForKey != 27 && capWebcam.isOpened()) {		// until the Esc key is pressed or webcam connection is lost
		if(getChannelFlag(charCheckForKey) != 99) {
			channelFlag = getChannelFlag(charCheckForKey);
		}
		bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);		// get next frame
		if (!blnFrameReadSuccessfully || imgOriginal.empty()) {		// if frame not read successfully
			std::cout << "error: frame not read from webcam\n";		// print error message to std out
			break;													// and jump out of while loop
		}
		ticks = (double)getTickCount();
		cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);

		if (trackModeFlag == 0) { // calibration mode
			getBoundingBoxHSV(imgHSV, BBOX, HSVMINALL[channelFlag], HSVMAXALL[channelFlag]);
			// bounding box	to show selected color region
			rectangle(imgOriginal,
				Point(BBOX[0], BBOX[1]),
				Point(BBOX[2], BBOX[3]),
				Scalar(200, 200, 200),
				1,
				8);
			putText(imgOriginal, "CAL", Point(30,30), FONT_HERSHEY_PLAIN , 1.5, Scalar(12, 12, 200), 2, 8, false); // indicate tracking mode

		} else if (trackModeFlag == 1) { // tracking mode
										 // do vision processing here
			detectCCBlobs(HSVMINALL, HSVMAXALL);
			putText(imgOriginal, "TRACK", Point(30,30), FONT_HERSHEY_PLAIN , 1.5, Scalar(12, 12, 200), 2, 8, false); // indicate tracking mode
		}

		imshow("imgOriginal", imgOriginal);			// show windows
		//imshow("imgThresh", imgThresh);
		ticks = ((double)getTickCount() - ticks)/getTickFrequency(); // time elapsed
		if (frameCount % 60 == 0) {
			printf("HSVMAX %d %d %d HSVMIN %d %d %d\n time %.3f\n ch %d\n", HSVMAXALL[channelFlag][0], HSVMAXALL[channelFlag][1], HSVMAXALL[channelFlag][2], HSVMINALL[channelFlag][0], HSVMINALL[channelFlag][1], HSVMINALL[channelFlag][2], ticks, channelFlag+1);
		}
		frameCount++;
		charCheckForKey = waitKey(1);			// delay (in ms) and get key press, if any
	}	// end while
	saveConfigFile(HSVMINALL,HSVMAXALL,3); // save HSV thresholds to config file
	return(0);
}


/// @brief Detect 2-channel color codes blobs over 3 channels
///
/// @param MINHSV
/// @param MAXHSV
///
/// @return Void
///
void detectCCBlobs(int MINHSV[][3], int MAXHSV[][3]) {

	// inRange, blur, eorde, dilate, etc
	// make a clone if required...findContours rewrites the original matrix
	//Mat imgThreshCopy = imgThresh.clone();
	// get the binary image
	cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	// ch1
	inRange(imgHSV, Scalar(MINHSV[0][0], MINHSV[0][1], MINHSV[0][2]), Scalar(MAXHSV[0][0], MAXHSV[0][1], MAXHSV[0][2]), imgThreshCh1);
	//GaussianBlur(imgThreshCh1, imgThreshCh1, cv::Size(3, 3), 0); // take out?
	erode(imgThreshCh1, imgThreshCh1, structuringElement);
	//dilate(imgThreshCh1, imgThreshCh1, structuringElement);
	// ch2
	inRange(imgHSV, Scalar(MINHSV[1][0], MINHSV[1][1], MINHSV[1][2]), Scalar(MAXHSV[1][0], MAXHSV[1][1], MAXHSV[1][2]), imgThreshCh2);
	//GaussianBlur(imgThreshCh2, imgThreshCh2, cv::Size(3, 3), 0); // take out?
	erode(imgThreshCh2, imgThreshCh2, structuringElement);
	//dilate(imgThreshCh2, imgThreshCh2, structuringElement);
	// ch3
	inRange(imgHSV, Scalar(MINHSV[2][0], MINHSV[2][1], MINHSV[2][2]), Scalar(MAXHSV[2][0], MAXHSV[2][1], MAXHSV[2][2]), imgThreshCh3);
	//GaussianBlur(imgThreshCh3, imgThreshCh3, cv::Size(3, 3), 0); // take out?
	erode(imgThreshCh3, imgThreshCh3, structuringElement);
	//dilate(imgThreshCh3, imgThreshCh3, structuringElement);

	int i;
	int dilateFactor = 35; // Amount to increase rect size by [%]
	vector<Rect> myFilteredRects1;
	vector<Rect> myFilteredRects2;
	vector<Rect> myFilteredRects3;
	Rect myCCRects[21];
	Scalar tmpColor = Scalar(255);
	Scalar transitColor = Scalar(0, 213, 255);
	Scalar ch1Color = Scalar(0, 213, 255);
	Scalar ch2Color = Scalar(181, 113, 220);
	Scalar ch3Color = Scalar(199, 220, 113);
	// get bounding rectangles from thresholded binary images
	getThresholdRects(imgThreshCh1, myFilteredRects1);
	getThresholdRects(imgThreshCh2, myFilteredRects2);
	getThresholdRects(imgThreshCh3, myFilteredRects3);

	// expand bounding rectangles
	dilateRects(dilateFactor, myFilteredRects1);
	dilateRects(dilateFactor, myFilteredRects2);
	dilateRects(dilateFactor, myFilteredRects3);

	// draw retangles for visualization
	for(i=0;i<myFilteredRects1.size();i++) {
		rectangle(imgOriginal, myFilteredRects1[i].tl(), myFilteredRects1[i].br(), ch1Color, 2, 8, 0); // bounding box
	}
	for(i=0;i<myFilteredRects2.size();i++) {
		rectangle(imgOriginal, myFilteredRects2[i].tl(), myFilteredRects2[i].br(), ch2Color, 2, 8, 0); // bounding box
	}
	for(i=0;i<myFilteredRects3.size();i++) {
		rectangle(imgOriginal, myFilteredRects3[i].tl(), myFilteredRects3[i].br(), ch3Color, 2, 8, 0); // bounding box
	}

	vector<int> usedRectsCh1(myFilteredRects1.size(),0); // keeping track of rectangles used already
	vector<int> usedRectsCh2(myFilteredRects2.size(),0);
	vector<int> usedRectsCh3(myFilteredRects3.size(),0);
	// find 2-color-code blobs
	if(getCCRectBinary(myFilteredRects1, myFilteredRects2, usedRectsCh1, usedRectsCh2, myCCRects, 0) == 0) {
		rectangle(imgOriginal, myCCRects[0].tl(), myCCRects[0].br(), tmpColor, 2, 8, 0); // CC blob
	}
	if(getCCRectBinary(myFilteredRects1, myFilteredRects3, usedRectsCh1, usedRectsCh3, myCCRects, 1) == 0) {
		rectangle(imgOriginal, myCCRects[1].tl(), myCCRects[1].br(), tmpColor, 2, 8, 0); // CC blob
	}
	if(getCCRectBinary(myFilteredRects2, myFilteredRects3, usedRectsCh2, usedRectsCh3, myCCRects, 2) == 0) {
		rectangle(imgOriginal, myCCRects[2].tl(), myCCRects[2].br(), tmpColor, 2, 8, 0); // CC blob
	}

}


/// @brief For a thresholded binary image get a vector of bounding rectangles
/// corresponding to the blobs
///
void getThresholdRects(Mat myImgThresh, vector<Rect> &filteredRect) {

	int i;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// findContours(imgThreshCh1, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0)); // get outermost contours
	findContours(myImgThresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0)); // get outermost contours

	vector<vector<Point> > contoursPoly(contours.size()); // to store approx polygonal curves
	vector<Rect> boundRect(contours.size()); // to store bounding rectangles  (x,y, at TL corner)
	for (i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contoursPoly[i], 3, true); // approx polygonal curve
		boundRect[i] = boundingRect(Mat(contoursPoly[i]));  // bounding box
		if (boundRect[i].area() > MINAREABLOB) {
			filteredRect.push_back(boundRect[i]); // append this rectangle to list of "good" blobs
		}
	}
}


/// @brief Dilate all the rectangles in a vector by a size percentage
///
/// Both the width and height are increased by the desired percentage.
///
/// @param Percentage factor to increase size [%]
/// @param myRect A vector of cv::rect rectangle objects
///
/// @return Void
///
void dilateRects(int factor, vector<Rect> &myRect) {
	int count = myRect.size();
	int i, dW, dH;
	for(i=0; i<count; i++) {
		dW = myRect[i].width*factor/100;
		dH = myRect[i].height*factor/100;
		myRect[i].width += dW; // increase width
		myRect[i].height += dH; // increase height
		myRect[i].x -= dW/2;// adjust x at TL
		myRect[i].y -= dH/2;// adjust x at TL
							// adjust y and TL
	}

}


/// @brief Find the 2-color-code rectangle which consists of two
/// different colored rectangles close to each other
///
/// The rectangles supplied have been enlarged slightly so there is
/// a slight overlap between the rectangles in close proximity. If
/// more than one pair are detected, then the pair with the largest
/// bounding rectangle by area is selected.
///
/// @param rectsChA vector of bounding rectangles of first channel
/// @param rectsChB vector of bounding rectangles of second channel
/// @param usedA vector indicating which elements have been allocated to a CC blob
/// @param usedB vector indicating which elements have been allocated to a CC blob
/// @param ccRects array all the two-color-code bounding rectangles
/// @param code ID of the current CC (color code)
///
/// @return 0 if successful, 1 if none detected
///
int getCCRectBinary(vector<Rect> &rectsChA, vector<Rect> &rectsChB, vector<int> &usedA, vector<int> &usedB, Rect ccRects[], int code)  {
	int i, j, iMax, jMax, iTarget, jTarget, maxArea=0;
	Rect tmpRect, selectedRect;
	iMax = rectsChA.size();
	jMax = rectsChB.size();
	for (i=0;i<iMax;i++) {
		for(j=0;j<jMax;j++) {
			if((usedA[i] == 0) && (usedB[j] == 0)) { // check if rects are unused
				tmpRect = rectsChA[i] & rectsChB[j]; // check for overlap
				if(tmpRect.area() > 0) {
					tmpRect =  rectsChA[i] | rectsChB[j]; // get union
					if (tmpRect.area()  > maxArea ) {
						selectedRect = tmpRect; // select if largest so far
						iTarget = i;
						jTarget = j;
						maxArea = tmpRect.area();
					}
				}
			}
		}
	}

	if(maxArea>0) {
		ccRects[code] = selectedRect;
		usedA[iTarget] = 1; // mark this element number as used
		usedB[jTarget] = 1;

		return 0;
	} else { return 1; }

}


static void onMouse(int event, int x, int y, int f, void*) {

	if (event == CV_EVENT_LBUTTONDOWN) {
		mouseDraggedFlag = 1;
		printf("bounding box top left (%d,%d)\n", x, y);
		BBOX[0] = x;
		BBOX[1] = y;
		BBOX[2] = x;
		BBOX[3] = y;
	}
	else if (event == CV_EVENT_MOUSEMOVE) {
		if (mouseDraggedFlag == 1) {
			printf("dragging (%d,%d)\n", x, y);
			BBOX[2] = x;
			BBOX[3] = y;
		}
	}
	else if (event == CV_EVENT_LBUTTONUP) {
		mouseDraggedFlag = 0;
	}
	else if (event == CV_EVENT_RBUTTONUP) {
		// toggle between calibrate and track modes
		if (trackModeFlag == 1) {
			trackModeFlag = 0;
		}
		else if (trackModeFlag == 0) {
			trackModeFlag = 1;
		}
	}
}


/// @brief given a char input from a keyboard, return
/// channel number corresponding to the channel index
///
/// @param charKey char key pressed
///
/// @return channel number
///
int getChannelFlag(char charKey) {

	if(charKey == '1') {
		return 0;
	} else if(charKey == '2') {
		return 1;
	} else if(charKey == '3') {
		return 2;
	} else {
		return 99; // default
	}

}


/// @brief Get minimum and maximum HSV values given a Mat object
/// and coordinates of the bounding box of interest.
///
/// 8-bit int HSV values are used
///
/// @param myImgObject HSV image to get HSV values from
/// @param BOX coordinates of the upper left and bottom right corners
/// of bounding box which indicates a color of interest [x1 y1 x2 y2]
/// [pixels]
/// @param MINHSV minimum HSV value from bounding box [H S V]
/// @param MAXHSV maximum HSV value from bounding box [H S V]
///
/// @return Void
///
void getBoundingBoxHSV(Mat myImgHSV, int BOX[], int MINHSV[], int MAXHSV[]) {

	int i, j, a;
	Vec3b intensity;
	// reset HSV values
	for (a = 0;a < 3;a++) {
		MAXHSV[a] = 0; MINHSV[a] = 255;
	}
	// get HSV range in selected region
	for (i = BOX[0];i < BOX[2];i++) {
		for (j = BOX[1];j < BOX[3];j++) {
			intensity = myImgHSV.at<Vec3b>(j, i);
			for (a = 0;a < 3;a++) {
				if (intensity.val[a] > MAXHSV[a]) { MAXHSV[a] = intensity.val[a]; }
				if (intensity.val[a] < MINHSV[a]) { MINHSV[a] = intensity.val[a]; }
			}
		}
	}
}


/// @brief Thresholding the image and detecting blobs
///
/// @param MINHSV minimum HSV value from bounding box [H S V]
/// @param MAXHSV maximum HSV value from bounding box [H S V]
///
/// @return Void
///
void detectBlobs(int MINHSV[], int MAXHSV[]) {
	// get the binary image
	inRange(imgHSV, Scalar(MINHSV[0], MINHSV[1], MINHSV[2]), Scalar(MAXHSV[0], MAXHSV[1], MAXHSV[2]), imgThresh);
	//inRange(imgHSV, Scalar(9,81,165), Scalar(14,154,229), imageThreshold); // skin color for quick testing
	//inRange(imgHSV, Scalar(0,92,255), Scalar(12,172,255), imageThreshold); // orange acrylic color for quick testing
	GaussianBlur(imgThresh, imgThresh, cv::Size(3, 3), 0); // take out?
	cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	erode(imgThresh, imgThresh, structuringElement);
	dilate(imgThresh, imgThresh, structuringElement);
	//dilate(imgThresh, imgThresh, structuringElement);

	RNG rng(12345);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	char TEXT[32];
	Mat imgThreshCopy = imgThresh.clone();
	findContours(imgThreshCopy, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0)); // get child contours
	vector<vector<Point> > contours_poly(contours.size()); // approx polygonal curve
	vector<Rect> boundRect(contours.size()); // this has x,y, at tl corner
	vector<Rect> filteredRect;
	filteredRect.clear();
	int nFilteredRect = 0;
	size_t i,j;
	int areaThresh = 0;

	for (i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true); // approx polygonal curve
		boundRect[i] = boundingRect(Mat(contours_poly[i]));  // bounding box
	}
	Mat drawing = Mat::zeros(imgThresh.size(), CV_8UC3);
	// draw contours and filter blobs by size
	for (i = 0; i< contours.size(); i++) {
		if (boundRect[i].area() > MINAREABLOB) {
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
			filteredRect.push_back(boundRect[i]); // append this rectangle to list of "good" blobs
												  // ADD CODE HERE TO COMBINE BLOBS CLOSE TO EACH OTHER
												  // see Rect_ http://docs.opencv.org/2.4/modules/core/doc/basic_structures.html
												  // rect1 & rect2
		}
	}
	// get largest area
	if(filteredRect.size() > 0) {
		for (i = 0; i< filteredRect.size(); i++) {
			if(filteredRect[i].area() > areaThresh) {
				areaThresh = filteredRect[i].area();
			}
		}
	}
	if(filteredRect.size() > 0) {
		// draw all, with two largest in a different color
		for (i = 0; i< filteredRect.size(); i++) {
			if (filteredRect[i].area() > (areaThresh*60/100) ) {
				Scalar color = Scalar(0,200,0);
				rectangle(imgOriginal, filteredRect[i].tl(), filteredRect[i].br(), color, 2, 8, 0); // bounding box
																									// draw angle line
			} else {
				Scalar color = Scalar(0,0,200);
				rectangle(imgOriginal, filteredRect[i].tl(), filteredRect[i].br(), color, 2, 8, 0); // bounding box
			}
		}
	}

	imshow("imgThresh", imgThresh);
	imshow("imgHSV", drawing);
}


/// @brief Load configuration file with HSV parameters
///
/// @param MIN array of arrays of minimum HSV threshold [H S V]
/// @param MAX array of arrays of maximum HSV threshold [H S V]
/// @param nChannels number of color channels in the config file
///
/// @return 0 if read successfully, 1 if failed
///
int loadConfigFile(int MIN[][3], int MAX[][3], int nChannels) {

	int i, ch;
	FILE *fp;
	char INPUTPATH[] = "babyMotionConfig.txt";
	fp = fopen(INPUTPATH,"r");
	if(fp==NULL){
		printf("read error!\n");
		return 1;
	}
	for(i=0;i<nChannels;i++) {
		fscanf(fp,"channel %d, HSVMIN{%d,%d,%d}, HSVMAX{%d,%d,%d}\n",&ch,&MIN[i][0],&MIN[i][1],&MIN[i][2],&MAX[i][0],&MAX[i][1],&MAX[i][2]);
	}
	fclose(fp);
	printf("done reading from config file!\n");
	return 0;
}


/// @brief Save configuration file with HSV parameters
///
/// @param MIN array of arrays of minimum HSV threshold [H S V]
/// @param MAX array of arrays of maximum HSV threshold [H S V]
/// @param nChannels number of color channels in the config file
///
/// @return 0 if read successfully, 1 if failed
///
int saveConfigFile(int MIN[][3], int MAX[][3], int nChannels) {

	int i;
	FILE *fp;
	char INPUTPATH[] = "babyMotionConfig.txt";
	fp = fopen(INPUTPATH,"w");
	if(fp==NULL){
		printf("config file write error!\n");
		return 1;
	}
	for(i=0;i<nChannels;i++) {
		fprintf(fp,"channel %d, HSVMIN{%d,%d,%d}, HSVMAX{%d,%d,%d}\n",i,MIN[i][0],MIN[i][1],MIN[i][2],MAX[i][0],MAX[i][1],MAX[i][2]);
	}
	fclose(fp);
	printf("done wiriting to config file!\n");
	return 0;
}
