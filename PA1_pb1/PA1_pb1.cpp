#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <cmath> 

#include <stdlib.h>

#include "ReadImage.cpp"
#include "WriteImage.cpp"

#define pi 3.141592653589793

using namespace cv;
using namespace std;

void generate_gaussian_kernel_1D(float* kernel, double sigma, int kernel_size);
void padd_with_zeros_1D(float* signal, float* padded_signal, int width, int filter_size);
void apply_gaussian_smoothing(float* signal, int signal_size, float kernel[], int kernel_size, float* output_signal);


void generate_gaussian_kernel_1D(float* kernel, double sigma, int kernel_size)
{
	int i;
	float cst, tssq, x, sum;

	cst = 1. / (sigma * sqrt(2.0 * pi));
	tssq = 1. / (2 * sigma * sigma);

	for (i = 0; i < kernel_size; i++)
	{
		x = (float)(i - kernel_size / 2);
		kernel[i] = (cst * exp(-(x * x * tssq)));
	}

	sum = 0.0;
	for (i = 0; i < kernel_size; i++)
		sum += kernel[i];

	for (i = 0; i < kernel_size; i++)
		kernel[i] /= sum;
}

void generate_gaussian_kernel_2D(float** kernel, double sigma, int kernel_size)
{
	int i, j;
	float cst, tssq, x, sum;

	cst = 1. / (sigma * sqrt(2.0 * pi));
	tssq = 1. / (2 * sigma * sigma);

	for (i = 0; i < kernel_size; i++)
	{
		for (j = 0; j < kernel_size; j++)
		{
			x = (float)(i - kernel_size / 2);
			kernel[i][j] = (cst * exp(-(x * x * tssq)));
		}
	}

	sum = 0.0;
	for (i = 0; i < kernel_size; i++)
		for (j = 0; j < kernel_size; j++)
			sum += kernel[i][j];

	for (i = 0; i < kernel_size; i++)
		for (j = 0; j < kernel_size; j++)
			kernel[i][j] /= sum;
}

void padd_with_zeros_1D(float* signal, float* padded_signal, int width, int filter_size)
{
	int i;
	int new_width = width + filter_size - 1;

	for (i = 0; i < new_width; i++)
	{
		padded_signal[i] = 0.0;
	}

	for (i = 0; i < width; i++)
	{
		padded_signal[i + (filter_size / 2)] = signal[i];
	}
}

void padd_with_zeros_2D(int** matrix, int** padded_matrix, int width, int height, int filter_size)
{
	int new_height = height + filter_size - 1;
	int new_width = width + filter_size - 1;

	for (int i = 0; i < new_height; i++)
		for (int j = 0; j < new_width; j++)
			padded_matrix[i][j] = 0;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			padded_matrix[i + (filter_size / 2)][j + (filter_size / 2)] = matrix[i][j];

}

void apply_gaussian_smoothing_1D(float* signal, int signal_size, float kernel[], int kernel_size, float* output_signal)
{
	float sum;
	for (int index = kernel_size / 2; index < signal_size - (kernel_size / 2); ++index)
	{
		sum = 0.0;
		for (int i = kernel_size / 2; i < kernel_size - (kernel_size / 2); ++i)
		{
			sum += signal[index + i] * kernel[i];
		}
		output_signal[index] = sum;
	}
	/*
	int index;
	float sum;
	for (index = 0; index < signal_size - kernel_size; index++)
	{
		sum = 0.0;
		for (int i = 0; i < kernel_size; i++)
		{
			sum += signal[index + i] * kernel[i];
		}
		output_signal[index] = sum;
	}
	*/
}

void apply_gaussian_smoothing_2D(int** image, int x_size, int y_size, float** kernel, int kernel_size, int** output_image)
{
	for (int index_i = kernel_size / 2; index_i < y_size - (kernel_size / 2); ++index_i)
	{
		for (int index_j = kernel_size / 2; index_j < x_size - (kernel_size / 2); ++index_j)
		{
			int sum = 0;

			//cout << index_i << " " << index_j << endl;
			for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i)
			{
				for (int j = -kernel_size / 2; j <= kernel_size / 2; ++j)
				{
					int data = image[index_i + i][index_j + j];
					float coeff = kernel[i + (kernel_size / 2)][j + (kernel_size / 2)];
					//int coeff = kernel[i + (kernel_size / 2)][j + (kernel_size / 2)];

					sum += data * coeff;
					//cout << "Data: " << data << "  Coeff: " << coeff << "        Sum: " << sum << endl;

				}
			}

			output_image[index_i][index_j] = sum;
		}
	}

	/*
	for (int i = 0; i < 20; i++) {
		for (int j = 0; j < 20; j++) {
			cout << output_image[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << "//////////////////////////////////////////////////////////////////////////////////" << endl;
	*/
}


int main()
{
	const float sigma = 1.0;
	const int mask_size = 5 * int(sigma);
	const int padded_size = 128 + mask_size - 1;


	/* 1D Gaussian Smoothing */
	// Read in Rect_128.txt as a matrix
	ifstream in("Rect_128.txt");
	float rect_128[128];
	for (int i = 0; i < 128; i++)
	{
		float f;
		in >> f;
		rect_128[i] = f;
	}
	cout << "Original Rect_128:  \t[ ";
	for (int i = 0; i < 128; i++)
		cout << rect_128[i] << " ";
	cout << "]" << endl << size(rect_128) << endl << endl;

	Mat orig_signal = Mat(1, 128, CV_32F, rect_128);
	Mat orig_signal_resized;
	resize(orig_signal, orig_signal_resized, cv::Size(640, 240), 0, 0, cv::INTER_AREA);
	imshow("Rect_128 (Original)", orig_signal_resized);
	imwrite("Rect_128 (Original).jpg", orig_signal_resized);

	// Pad signal with zeros
	float rect_128_padded[132];
	padd_with_zeros_1D(rect_128, rect_128_padded, 128, 5);
	cout << "Padded Rect_128:  \t[ ";
	for (int i = 0; i < 132; i++)
		cout << rect_128_padded[i] << " ";
	cout << "]" << endl << size(rect_128_padded) << endl << endl;

	// Create Gaussian mask 
	float Gaussian_Kernel_1D[mask_size];
	generate_gaussian_kernel_1D(Gaussian_Kernel_1D, sigma, mask_size);
	cout << "1D Gaussian Kernel:  \t[ ";
	for (int i = 0; i < mask_size; i++)
		cout << Gaussian_Kernel_1D[i] << " ";
	cout << "]" << endl << size(Gaussian_Kernel_1D) << endl << endl;

	// Apply Gaussian smoothing to signal
	float rect_128_smoothed[128];
	apply_gaussian_smoothing_1D(rect_128_padded, 128, Gaussian_Kernel_1D, mask_size, rect_128_smoothed);

	// Display smoothed signal
	cout << "Smoothed Rect_128:  \t[ ";
	for (int i = 0; i < 128 - mask_size; i++)
		cout << rect_128_smoothed[i] << " ";
	cout << "]" << endl << size(rect_128_smoothed) << endl << endl;

	Mat smoothed_signal = Mat(1, 128, CV_32F, rect_128_smoothed);
	Mat smoothed_signal_resized;
	resize(smoothed_signal, smoothed_signal_resized, cv::Size(640, 240), 0, 0, cv::INTER_AREA);
	imshow("Rect_128 (Smoothed)", smoothed_signal_resized);
	imwrite("Rect_128 (Smoothed).jpg", smoothed_signal_resized);


	//////////////////////////////////////////////////////
	/* 2D Gaussian Smooting */
	int** input, ** output;
	int x_size, y_size, Q;
	char name[20] = "lenna.pgm";
	char outfile[20] = "lenna_smoothed.pgm";


	ReadImage(name, &input, x_size, y_size, Q);

	cout << endl << "Original Image:  " << endl;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			cout << input[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;


	// Generate 2D Gaussian kernel
	float** Gaussian_Kernel_2D;
	Gaussian_Kernel_2D = new float* [mask_size];
	for (int i = 0; i < mask_size; i++)
		Gaussian_Kernel_2D[i] = new float[mask_size];
	generate_gaussian_kernel_2D(Gaussian_Kernel_2D, sigma, mask_size);

	cout << "2D Gaussian Kernel:  " << endl;
	for (int i = 0; i < mask_size; i++)
	{
		for (int j = 0; j < mask_size; j++)
		{
			cout << Gaussian_Kernel_2D[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	// Pad image with zeros
	int** input_padded;
	input_padded = new int* [y_size + mask_size - 1];
	for (int i = 0; i < y_size + mask_size - 1; i++)
		input_padded[i] = new int[x_size + mask_size - 1];
	padd_with_zeros_2D(input, input_padded, x_size, y_size, mask_size);

	cout << "Padded Image:  " << endl;
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			cout << input_padded[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	// Apply Gaussian smoothing to image
	int** input_smoothed;
	input_smoothed = new int* [y_size + mask_size - 1];
	for (int i = 0; i < y_size + mask_size - 1; i++)
		input_smoothed[i] = new int[x_size + mask_size - 1];

	for (int i = 0; i < y_size + mask_size - 1; i++)
		for (int j = 0; j < x_size + mask_size - 1; j++)
			input_smoothed[i][j] = 0;
	//apply_gaussian_smoothing_2D(input_padded, x_size + mask_size - 1, y_size + mask_size - 1, Gaussian_Kernel_2D, mask_size, input_smoothed);
	apply_gaussian_smoothing_2D(input_padded, x_size, y_size, Gaussian_Kernel_2D, mask_size, input_smoothed);

	cout << "Smoothed Image:  " << endl;
	for (int i = 0; i < 20; i++) {
		for (int j = 0; j < 20; j++) {
			cout << input_smoothed[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;


	WriteImage(outfile, input_smoothed, x_size, y_size, Q);


	waitKey(0);
	return 0;
}