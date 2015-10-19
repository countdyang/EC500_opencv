#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int, char**)
{
	Mat gray1 = imread("1.jpg",0);
	namedWindow("Gray1", 1);    
    imshow("Gray1", gray1);

	Mat gray2 = imread("taoyang2.jpg", 0);
	namedWindow("Gray2", 1);
	imshow("Gray2", gray2);

	// Initialize parameters
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	// Calculate histogram
	MatND hist1;
	calcHist(&gray1, 1, 0, Mat(), hist1, 1, &histSize, ranges, true, false);

	MatND hist2;
	calcHist(&gray2, 1, 0, Mat(), hist2, 1, &histSize, ranges, true, false);

	for (int i = 0; i < 4; i++)
	{
		int compare_method = i;
		double base_base = compareHist(hist1, hist1, compare_method);
		//double base_half = compareHist(hist1, hist2, compare_method);
		double base_test1 = compareHist(hist1, hist2, compare_method);
		//double base_test2 = compareHist(hist1, hist2, compare_method);

		printf(" Method [%d] Perfect, Base-Test(1) : %f,%f \n", i, base_base,  base_test1);
	}

	printf("Done \n");


	// Show the calculated histogram in command window
	double total1;
	total1 = gray1.rows * gray1.cols;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = hist1.at<float>(h);
		cout << " " << binVal;
	}

	double total2;
	total2 = gray2.rows * gray2.cols;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = hist2.at<float>(h);
		cout << " " << binVal;
	}

	// Plot the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage1(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(hist1, hist1, 0, histImage1.rows, NORM_MINMAX, -1, Mat());

	Mat histImage2(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(hist2, hist2, 0, histImage2.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage1, Point(bin_w*(i - 1), hist_h - cvRound(hist1.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist1.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	for (int i = 1; i < histSize; i++)
	{
		line(histImage2, Point(bin_w*(i - 1), hist_h - cvRound(hist2.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist2.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	namedWindow("Result1", 1);    imshow("Result1", histImage1);
	namedWindow("Result2", 1);    imshow("Result2", histImage2);

	waitKey(0);
	return 0;
}
