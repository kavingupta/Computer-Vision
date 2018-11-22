

//PRE Compiled Headers
#include "pch.h"
#include "opencv2\opencv.hpp"

using namespace cv;
using namespace std;
int match_method;
int max_Trackbar = 5;



int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;


/// Global Variables
Mat src1; Mat hsv; Mat hue;
int bins = 25;

/// Function Headers
void Hist_and_Backproj(int, void*);


/**
* @function Hist_and_Backproj
* @brief Callback to Trackbar
*/
void Hist_and_Backproj(int, void*)
{
	MatND hist;
	int histSize = MAX(bins, 2);
	float hue_range[] = { 0, 180 };
	const float* ranges = { hue_range };

	/// Get the Histogram and normalize it
	calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

	/// Get Backprojection
	MatND backproj;
	calcBackProject(&hue, 1, 0, hist, backproj, &ranges, 1, true);

	/// Draw the backproj
	//imshow("BackProj", backproj);

	/// Draw the histogram
	int w = 400; int h = 400;
	int bin_w = cvRound((double)w / histSize);
	Mat histImg = Mat::zeros(w, h, CV_8UC3);

	for (int i = 0; i < bins; i++)
	{
		rectangle(histImg, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(hist.at<float>(i)*h / 255.0)), Scalar(0, 0, 255), -1);
	}

	//imshow("Histogram", histImg);
}
vector<Rect> traffic_objects, red_objects, green_objects, yellow_objects;
struct trafficLight {
public:
	Rect bounding_rect;
	char* state;
};
vector<struct trafficLight> detected_lights;
Mat getGreenImg(Mat img)
{
	Mat hsv, maskImg;
	cvtColor(img, hsv, CV_BGR2HSV);
	//inRange(hsv, Scalar(55, 16, 206), Scalar(91, 255, 255), maskImg);
	inRange(hsv, Scalar(70, 90, 200), Scalar(100, 160, 255), maskImg);
	dilate(maskImg, maskImg, Mat(), Point(-1, -1), 5);

	Mat resultImg = Mat(img.size(), CV_8UC1);
	resultImg = Scalar::all(0);

	Mat contourImg = maskImg.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(contourImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	// filter contours
	if (contours.size() > 0)
	{
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
		{
			Rect rect = boundingRect(contours[idx]);
			rectangle(resultImg, rect, Scalar(255), -1);
			//rectangle(img, rect, Scalar(255, 0, 0), -1);
			green_objects.push_back(rect);
		}
	}

	return resultImg;
}

Mat getRedImg(Mat img)
{
	Mat hsv, maskImg;
	cvtColor(img, hsv, CV_BGR2HSV);
	inRange(hsv, Scalar(0, 200, 240), Scalar(20, 240, 255), maskImg);
	dilate(maskImg, maskImg, Mat(), Point(-1, -1), 5);

	Mat resultImg = Mat(img.size(), CV_8UC1);
	resultImg = Scalar::all(0);

	Mat contourImg = maskImg.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(contourImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	// filter contours
	if (contours.size() > 0)
	{
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
		{
			Rect rect = boundingRect(contours[idx]);
			rectangle(resultImg, rect, Scalar(255), -1);
			//rectangle(img, rect, Scalar(0, 0, 255), -1);
			red_objects.push_back(rect);
		}
	}

	return resultImg;
}

Mat getYellowImg(Mat img)
{
	Mat hsv, maskImg;
	cvtColor(img, hsv, CV_BGR2HSV);
	inRange(hsv, Scalar(15, 170, 230), Scalar(40, 230, 255), maskImg);
	dilate(maskImg, maskImg, Mat(), Point(-1, -1), 5);

	Mat resultImg = Mat(img.size(), CV_8UC1);
	resultImg = Scalar::all(0);

	Mat contourImg = maskImg.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(contourImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	// filter contours
	if (contours.size() > 0)
	{
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
		{
			Rect rect = boundingRect(contours[idx]);
			rectangle(resultImg, rect, Scalar(255), -1);
			//rectangle(img, rect, Scalar(0, 255, 255), -1);
			yellow_objects.push_back(rect);
		}
	}

	return resultImg;
}

Mat getTrafficCandidate(Mat img)
{
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	Mat resultImg = Mat(img.size(), CV_8UC1);
	resultImg = Scalar::all(0);

	// morphological gradient
	Mat grad;
	Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(gray, grad, MORPH_GRADIENT, morphKernel);

	// binarize
	Mat bw;
	threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);

	// connect horizontally oriented regions
	Mat connected;
	morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
	morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
	threshold(connected, connected, 0, 255, THRESH_BINARY_INV);
	erode(connected, connected, Mat(), Point(-1, -1), 2);

	Mat contourImg = connected.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(contourImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	// filter contours
	if (contours.size() > 0)
	{
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
		{
			Rect rect = boundingRect(contours[idx]);
			float width_ratio = float(connected.rows) / float(rect.width);
			if (width_ratio > 10)
			{
				float height_ratio = float(connected.cols) / float(rect.height);
				if (height_ratio > 5)
				{
					float ratio = float(rect.height) / float(rect.width);
					if (ratio > 1 && ratio < 4)
					{
						rectangle(resultImg, rect, Scalar(255), -1);
						//rectangle(img, rect, Scalar(255, 255, 0), 1);		
						traffic_objects.push_back(rect);
					}
				}
			}
		}
	}

	//dilate(resultImg, resultImg, Mat(), Point(-1, -1), 5);
	return resultImg;
}

vector<struct trafficLight> getTrafficLight(char* szFileName)
{
	detected_lights.clear();
	Mat img = imread(szFileName);
	bool bGreen = false, bRed = false, bYellow = false;

	Mat trafficImg = getTrafficCandidate(img);

	Mat redImg = getRedImg(img);
	Mat yellowImg = getYellowImg(img);
	Mat greenImg = getGreenImg(img);

	Mat finalImg;
	bitwise_or(trafficImg, greenImg, finalImg);
	bitwise_or(redImg, finalImg, finalImg);
	bitwise_or(yellowImg, finalImg, finalImg);

	bGreen = false;
	bRed = false;
	bYellow = false;
	Mat contourImg = finalImg.clone();
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(contourImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	// filter contours
	struct trafficLight currLight;
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{

		Rect rect = boundingRect(contours[idx]);
		float width_ratio = float(finalImg.rows) / float(rect.width);
		if (width_ratio > 5 && width_ratio < 80)
		{
			float height_ratio = float(finalImg.cols) / float(rect.height);
			if (height_ratio > 5 && height_ratio < 40)
			{
				float ratio = float(rect.height) / float(rect.width);
				if (ratio > 1.5 && ratio < 4)
				{
					bool btraffic = false;
					for (int traffic_idx = 0; traffic_idx < traffic_objects.size(); traffic_idx++)
					{
						int traffic_center_x = traffic_objects[traffic_idx].x + traffic_objects[traffic_idx].width / 2;
						int traffic_center_y = traffic_objects[traffic_idx].y + traffic_objects[traffic_idx].height / 2;
						if (traffic_center_x > rect.x && traffic_center_x < rect.x + rect.width)
						{
							if (traffic_center_y > rect.y && traffic_center_y < rect.x + rect.height)
							{
								btraffic = true;
								break;
							}
						}
					}

					if (btraffic)
					{
						currLight.bounding_rect = rect;
						for (int yellow_idx = 0; yellow_idx < yellow_objects.size(); yellow_idx++)
						{
							int yellow_center_x = yellow_objects[yellow_idx].x + yellow_objects[yellow_idx].width / 2;
							int yellow_center_y = yellow_objects[yellow_idx].y + yellow_objects[yellow_idx].height / 2;
							if (yellow_center_x > rect.x && yellow_center_x < rect.x + rect.width)
							{
								if (yellow_center_y > rect.y && yellow_center_y < rect.y + rect.height)
								{
									rectangle(img, rect, Scalar(0, 255, 255), 2);
									bYellow = true;
									break;
								}
							}
						}

						if (bYellow) {
							currLight.state = "Amber";
							break;
						}
						else
						{
							for (int green_idx = 0; green_idx < green_objects.size(); green_idx++)
							{
								int green_center_x = green_objects[green_idx].x + green_objects[green_idx].width / 2;
								int green_center_y = green_objects[green_idx].y + green_objects[green_idx].height / 2;
								if (green_center_x > rect.x && green_center_x < rect.x + rect.width)
								{
									if (green_center_y > rect.y + rect.height * 2 / 3 && green_center_y < rect.y + rect.height)
									{
										rectangle(img, rect, Scalar(255, 0, 0), 2);
										bGreen = true;
										currLight.state = "Green";
										break;
									}
								}
							}

							if (!bGreen)
							{
								for (int red_idx = 0; red_idx < red_objects.size(); red_idx++)
								{
									int red_center_x = red_objects[red_idx].x + red_objects[red_idx].width / 2;
									int red_center_y = red_objects[red_idx].y + red_objects[red_idx].height / 2;
									if (red_center_x > rect.x && red_center_x < rect.x + rect.width)
									{
										if (red_center_y > rect.y && red_center_y < rect.y + rect.height / 3)
										{
											rectangle(img, rect, Scalar(0, 0, 255), 2);
											bRed = true;
											currLight.state = "Red";
											break;
										}
									}
								}
							}
						}
					}
				}
			}
		}
		detected_lights.push_back(currLight);
	}
	if (bYellow)
		putText(img, "Amber", Point(20, 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 255, 255), 8);
	else if (bRed)
		putText(img, "Red", Point(20, 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 8);
	else if (bGreen)
		putText(img, "Green", Point(20, 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 255, 0), 8);
	else
	{
		if (green_objects.size() > 0)
			putText(img, "Green", Point(20, 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 255, 0), 8);
		else if (red_objects.size() > 0)
			putText(img, "Red", Point(20, 50), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 8);
	}

	imshow("input", img);
	waitKey(0);

	return img;
}

int main(int argc, char* argv[])
{

	//BLOB DETECTION 

	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 150;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 150;

	// Filter by Circularity
	//params.filterByCircularity = true;
	//params.minCircularity = 0.1;

	// Filter by Convexity
	//params.filterByConvexity = true;
	//params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.02;


	// Storage for blobs
	vector<KeyPoint> keypoints;


#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

	// Set up detector with params
	SimpleBlobDetector detector(params);

	// Detect blobs
	detector.detect(im, keypoints);
#else 

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Detect blobs
	//detector->detect(img1, keypoints);
#endif 

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
	// the size of the circle corresponds to the size of blob

	Mat im_with_keypoints;
	//drawKeypoints(im, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//drawKeypoints(img1, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	//	imshow("keypoints", im_with_keypoints);



	char* images[] = {
		"test/CamVidLights01.png",
		"test/CamVidLights02.png",
		"test/CamVidLights03.png",
		"test/CamVidLights04.png",
		"test/CamVidLights05.png",
		"test/CamVidLights06.png",
		"test/CamVidLights07.png",
		"test/CamVidLights08.png",
		"test/CamVidLights09.png",
		"test/CamVidLights10.png",
		"test/CamVidLights11.png",
		"test/CamVidLights12.png",
		"test/CamVidLights13.png",
		"test/CamVidLights14.png",
	};
	int truth_boxes[] = {
		319,202,346,279,
		692,264,711,322,
		217,103,261,230,
		794,212,820,294,
		347,210,373,287,
		701,259,720,318,
		271,65,309,189,
		640,260,652,301,
		261,61,302,193,
		644,269,657,312,
		238,42,284,187,
		650,279,663,323,
		307,231,328,297,
		747,266,764,321,
		280,216,305,296,
		795,253,816,316,
		359,246,380,305,
		630,279,646,327,
		260,122,299,239,
		691,271,705,315,
		331,260,349,312,
		663,280,676,322,
		373,219,394,279,
		715,242,732,299,
		423,316,429,329,
		516,312,521,328,
		283,211,299,261,
		604,233,620,279,
		294,188,315,253,
		719,225,740,286,
	};

	char* states[] = { "Green","Green","Green","Green","Green","Green","Red","Red","Red+Amber","Red+Amber","Green","Green","Amber","Amber","Amber","Amber","Green"
		,"Green","Green","Green","Green","Green","Green","Green","Red","Red","Red","Red","Red","Red" };

	int fp = 0;
	int fn = 0;
	int tn = 0;
	int tp = 0;
	int correct_state = 0;
	double dice_coeff = 0;

	double precision = 0, accuracy = 0, recall = 0;

	for (int i = 0; i < 1; i++) {
		vector<struct trafficLight> final_rects = getTrafficLight(images[i]);
		int num_traffic_lights = 0;
		if (i < 11) {
			Rect true_rect_1 = Rect(truth_boxes[8 * i], truth_boxes[8 * i + 1], truth_boxes[8 * i + 2] - truth_boxes[8 * i],
				truth_boxes[8 * i + 3] - truth_boxes[8 * i + 1]);
			Rect true_rect_2 = Rect(truth_boxes[8 * i + 4], truth_boxes[8 * i + 5],
				truth_boxes[8 * i + 6] - truth_boxes[8 * i + 4], truth_boxes[8 * i + 7] - truth_boxes[8 * i + 5]);

			for (int j = 0; j < final_rects.size(); j++) {
				char* my_state = final_rects[j].state;
				Rect my_rect = final_rects[j].bounding_rect;
				if (((my_rect & true_rect_1).area() / true_rect_1.area() >= 0.8) && (abs(my_rect.area() - true_rect_1.area()) <= 0.2*true_rect_1.area())) {
					++tp;
					++num_traffic_lights;
					dice_coeff += (double)((my_rect & true_rect_1).area() * 2) / (double)(my_rect.area() + true_rect_1.area());
					if (my_state != NULL && (my_state == states[2 * i]))
						correct_state++;
					break;
				}
				else if (((my_rect & true_rect_2).area() / true_rect_2.area() >= 0.8) && (abs(my_rect.area() - true_rect_2.area()) <= 0.2*true_rect_2.area())) {
					++tp;
					++num_traffic_lights;
					dice_coeff += (double)((my_rect & true_rect_2).area() * 2) / (double)(my_rect.area() + true_rect_2.area());
					if (my_state != NULL && (my_state == states[2 * i + 1]))
						correct_state++;
					break;
				}
				else {
					++fp;
				}

			}
			fn += 2 - num_traffic_lights;
		}

		else if (i == 11) {
			Rect true_rect_1 = Rect(truth_boxes[8 * i], truth_boxes[8 * i + 1], truth_boxes[8 * i + 2] - truth_boxes[8 * i],
				truth_boxes[8 * i + 3] - truth_boxes[8 * i + 1]);
			Rect true_rect_2 = Rect(truth_boxes[8 * i + 4], truth_boxes[8 * i + 5],
				truth_boxes[8 * i + 6] - truth_boxes[8 * i + 4], truth_boxes[8 * i + 7] - truth_boxes[8 * i + 5]);
			Rect true_rect_3 = Rect(truth_boxes[8 * i + 8], truth_boxes[8 * i + 9],
				truth_boxes[8 * i + 10] - truth_boxes[8 * i + 8], truth_boxes[8 * i + 11] - truth_boxes[8 * i + 9]);
			Rect true_rect_4 = Rect(truth_boxes[8 * i + 12], truth_boxes[8 * i + 13],
				truth_boxes[8 * i + 14] - truth_boxes[8 * i + 12], truth_boxes[8 * i + 15] - truth_boxes[8 * i + 13]);

			for (int j = 0; j < final_rects.size(); j++) {
				char* my_state = final_rects[j].state;
				Rect my_rect = final_rects[j].bounding_rect;
				if (((my_rect & true_rect_1).area() / true_rect_1.area() >= 0.8) && (abs(my_rect.area() - true_rect_1.area()) <= 0.2*true_rect_1.area())) {
					++tp;
					++num_traffic_lights;
					dice_coeff += (double)((my_rect & true_rect_1).area() * 2) / (double)(my_rect.area() + true_rect_1.area());
					if (my_state != NULL && (my_state == states[2 * i]))
						correct_state++;
					break;
				}
				else if (((my_rect & true_rect_2).area() / true_rect_2.area() >= 0.8) && (abs(my_rect.area() - true_rect_2.area()) <= 0.2*true_rect_2.area())) {
					++tp;
					++num_traffic_lights;
					dice_coeff += (double)((my_rect & true_rect_2).area() * 2) / (double)(my_rect.area() + true_rect_2.area());
					if (my_state != NULL && (my_state == states[2 * i + 1]))
						correct_state++;
					break;
				}
				else if (((my_rect & true_rect_3).area() / true_rect_3.area() >= 0.8) && (abs(my_rect.area() - true_rect_3.area()) <= 0.2*true_rect_3.area())) {
					++tp;
					++num_traffic_lights;
					dice_coeff += (double)((my_rect & true_rect_3).area() * 2) / (double)(my_rect.area() + true_rect_3.area());
					if (my_state != NULL && (my_state == states[2 * i + 2]))
						correct_state++;
					break;
				}
				else if (((my_rect & true_rect_4).area() / true_rect_4.area() >= 0.8) && (abs(my_rect.area() - true_rect_4.area()) <= 0.2*true_rect_4.area())) {
					++tp;
					++num_traffic_lights;
					dice_coeff += (double)((my_rect & true_rect_4).area() * 2) / (my_rect.area() + true_rect_4.area());
					if (my_state != NULL && (my_state == states[2 * i + 3]))
						correct_state++;
					break;
				}
				else {
					++fp;
				}
			}
			fn += 4 - num_traffic_lights;
		}
		else {
			Rect true_rect_1 = Rect(truth_boxes[8 * i + 8], truth_boxes[8 * i + 9], truth_boxes[8 * i + 10] - truth_boxes[8 * i + 8],
				truth_boxes[8 * i + 11] - truth_boxes[8 * i + 9]);
			Rect true_rect_2 = Rect(truth_boxes[8 * i + 12], truth_boxes[8 * i + 13],
				truth_boxes[8 * i + 14] - truth_boxes[8 * i + 12], truth_boxes[8 * i + 15] - truth_boxes[8 * i + 13]);

			for (int j = 0; j < final_rects.size(); j++) {
				char* my_state = final_rects[j].state;
				Rect my_rect = final_rects[j].bounding_rect;

				if (((my_rect & true_rect_1).area() / true_rect_1.area() >= 0.8) && (abs(my_rect.area() - true_rect_1.area()) <= 0.2*true_rect_1.area())) {
					++tp;
					++num_traffic_lights;
					dice_coeff += (double)((my_rect & true_rect_1).area() * 2) / (double)(my_rect.area() + true_rect_1.area());
					if (my_state != NULL && (my_state == states[2 * i + 2]))
						correct_state++;
					break;
				}
				else if (((my_rect & true_rect_2).area() / true_rect_2.area() >= 0.8) && (abs(my_rect.area() - true_rect_2.area()) <= 0.2*true_rect_2.area())) {
					++tp;
					++num_traffic_lights;
					dice_coeff += (double)((my_rect & true_rect_2).area() * 2) / (double)(my_rect.area() + true_rect_2.area());
					if (my_state != NULL && (my_state == states[2 * i + 3]))
						correct_state++;
					break;
				}
				else {
					++fp;
				}

			}
			fn += 2 - num_traffic_lights;
		}


	}

	recall = (double)tp / ((double)tp + (double)fn);
	precision = (double)tp / ((double)tp + (double)fp);
	accuracy = (double)tp / 30;
	dice_coeff = dice_coeff / (double)30;

	cout << "Accuracy: " << accuracy << endl;
	cout << "Recall: " << recall << endl;
	cout << "Precision: " << precision << endl;

	cout << "Percentage of lights with correct state: " << ((double)correct_state / 30) * 100 << "%" << endl;
	cout << "Dice Coefficient: " << dice_coeff / 30 << endl;
	waitKey(0);
	return 0;


}


void MatchingMethod(int, void*)
{
	/// Source image to display
	Mat img_display;

	string s;




	//result.create(result_rows, result_cols, CV_32FC1);
	//CannyThreshold(0, 0);

	/// Do the Matching and Normalize
	//matchTemplate(img, templ, result, match_method);
	//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	//minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// Show me what you got
	//rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2, 8, 0);
	//rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2, 8, 0);

	/*imshow(image_window, img_display);
	imshow(result_window, result);*/
}






