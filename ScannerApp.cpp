#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat capture, imgGray, imgBlur, imgCanny, imgThre, imgDil, imgErode, imgWarp, imgCrop;
vector<Point> unprocessedPoints, docPoints;
float w = 420, h = 596;

Mat preProcessing(Mat img)
{
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernel);
	erode(imgDil, imgErode, kernel);
	return imgDil;
}

vector<Point> getPaperContours(Mat image) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	vector<Point> paperCountour;
	int maxArea = 0;

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		string objectType;

		if (area > 1000)
		{
			float perimeter = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * perimeter, true);

			if (area > maxArea && conPoly[i].size() == 4) {
				paperCountour = { conPoly[i][0],conPoly[i][1] ,conPoly[i][2] ,conPoly[i][3] };
				maxArea = area;
			}
		}
	}
	return paperCountour;
}


vector<Point> rearrangeCorners(vector<Point> points)
{
	vector<Point> newCorners;
	vector<int>  addedPoints, substractedPoints;

	for (int i = 0; i < 4; i++)
	{
		addedPoints.push_back(points[i].x + points[i].y);
		substractedPoints.push_back(points[i].x - points[i].y);
	}

	newCorners.push_back(points[min_element(addedPoints.begin(), addedPoints.end()) - addedPoints.begin()]); 
	newCorners.push_back(points[max_element(substractedPoints.begin(), substractedPoints.end()) - substractedPoints.begin()]); 
	newCorners.push_back(points[min_element(substractedPoints.begin(), substractedPoints.end()) - substractedPoints.begin()]); 
	newCorners.push_back(points[max_element(addedPoints.begin(), addedPoints.end()) - addedPoints.begin()]); 
	return newCorners;
}

int captureImage()
{
	VideoCapture cap(0);

	Mat save_img;

	cap >> save_img;

	char Esc = 0;

	while (Esc != 27 && cap.isOpened()) {
		bool Frame = cap.read(save_img);
		if (!Frame || save_img.empty()) {
			cout << "error: frame not read from webcam\n";
			break;
		}
		namedWindow("save_image", cv::WINDOW_NORMAL);
		imshow("paper", save_img);
		Esc = waitKey(1);
	}
	imwrite("Images/paper.jpg", save_img);
	return 0;
}

Mat getWarp(Mat img, vector<Point> points, float w, float h)
{
	Point2f src[4] = { points[0],points[1],points[2],points[3] };
	Point2f dst[4] = { {0.0f,0.0f},{w,0.0f},{0.0f,h},{w,h} };

	Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, imgWarp, matrix, Point(w, h));

	return imgWarp;
}

void main() {
	captureImage();
	string path = "Images/paper.jpg";
	capture = imread(path);
	resize(capture, capture, Size(), 0.5, 0.5);

	imgThre = preProcessing(capture);
	unprocessedPoints = getPaperContours(imgThre);
	docPoints = rearrangeCorners(unprocessedPoints);

	imgWarp = getWarp(capture, docPoints, w, h);
	int cropVal = 5;
	Rect roi(cropVal, cropVal, w - (2 * cropVal), h - (2 * cropVal));
	imgCrop = imgWarp(roi);

	imshow("Image", capture);

	imshow("Image Crop", imgCrop);
	waitKey(0);

}