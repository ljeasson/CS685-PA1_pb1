//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>

#include <stdlib.h>

#include "ReadImage.cpp"
#include "WriteImage.cpp"

#define pi 3.141592653589793

//using namespace cv;
using namespace std;

void generate_gaussian_kernel_1D(float* kernel, double sigma, int kernel_size);
void generate_gaussian_kernel_2D(float** kernel, double sigma, int kernel_size);
void padd_with_zeros_1D(float* signal, float* padded_signal, int width, int filter_size);
void padd_with_zeros_2D(int** matrix, int** padded_matrix, int width, int height, int filter_size);
void apply_gaussian_smoothing_1D(float* signal, int signal_size, float kernel[], int kernel_size, float* output_signal);
void apply_gaussian_smoothing_2D(int** image, int x_size, int y_size, float** kernel, int kernel_size, int** output_image);
void apply_gaussian_smoothing_2D_with_1D(int** image, int x_size, int y_size, float kernel[], int kernel_size, int** output_image);


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
	float min_value = 0;
	float max_value = 0;
	for (int index = kernel_size / 2; index < signal_size - (kernel_size / 2); ++index)
	{
		float sum = 0.0;
		for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i)
		{
			float data = signal[index + i];
			float coeff = kernel[i + (kernel_size / 2)];
			
			sum += data * coeff;

			if (sum < min_value)
				min_value = sum;
			if (sum > max_value)
				max_value = sum;
		}
		output_signal[index - kernel_size / 2] = sum;
	}
	if (min_value < 0) min_value = 0;

	for (int index = kernel_size / 2; index < signal_size - (kernel_size / 2); ++index)
	{
		float value = output_signal[index - kernel_size / 2];
		output_signal[index - kernel_size / 2] = (value - min_value) / (max_value - min_value);
	}
}

void apply_gaussian_smoothing_2D(int** image, int x_size, int y_size, float** kernel, int kernel_size, int** output_image)
{
	float min_value = 0;
	float max_value = 0;
	for (int index_i = kernel_size / 2; index_i < y_size - (kernel_size / 2); ++index_i)
	{
		for (int index_j = kernel_size / 2; index_j < x_size - (kernel_size / 2); ++index_j)
		{
			float sum = 0;

			for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i)
			{
				for (int j = -kernel_size / 2; j <= kernel_size / 2; ++j)
				{
					float data = image[index_i + i][index_j + j];
					float coeff = kernel[i + (kernel_size / 2)][j + (kernel_size / 2)];

					sum += data * coeff;

					if (sum < min_value)
						min_value = sum;
					if (sum > max_value)
						max_value = sum;
				}
			}

			output_image[index_i - kernel_size/2][index_j - kernel_size/2] = sum;
		}
	}

	for (int index_i = kernel_size / 2; index_i < y_size - (kernel_size / 2); ++index_i)
	{
		for (int index_j = kernel_size / 2; index_j < x_size - (kernel_size / 2); ++index_j)
		{
			int value = output_image[index_i - kernel_size / 2][index_j - kernel_size / 2];
			output_image[index_i - kernel_size / 2][index_j - kernel_size / 2] = 255 * (value - min_value) / (max_value - min_value);

		}
	}
}

void apply_gaussian_smoothing_2D_with_1D(int** image, int x_size, int y_size, float kernel[], int kernel_size, int** output_image)
{
	float min_value;
	float max_value;
	
	// Perform convolution on x-axis
	min_value = 0;
	max_value = 0;
	for (int index_i = kernel_size / 2; index_i < y_size - (kernel_size / 2); ++index_i)
	{
		for (int index_j = kernel_size / 2; index_j < x_size - (kernel_size / 2); ++index_j)
		{
			float sum = 0.0;
			for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i)
			{
				float data = image[index_i][index_j + i];
				float coeff = kernel[i + (kernel_size / 2)];

				sum += data * coeff;

				//cout << data << "\t" << coeff << "\t" << sum << endl;

				if (sum < min_value)
					min_value = sum;
				if (sum > max_value)
					max_value = sum;
			}
			output_image[index_i - kernel_size / 2][index_j - kernel_size / 2] = sum;
			//cout << sum << endl;
			//cout << output_image[index_i - kernel_size / 2][index_j - kernel_size / 2] << endl << endl;
		}
	}
	//cout << min_value << endl << max_value << endl;

	// Perform convolution on x-axis
	min_value = 0;
	max_value = 0;
	for (int index_i = kernel_size / 2; index_i < y_size - (kernel_size / 2); ++index_i)
	{
		for (int index_j = kernel_size / 2; index_j < x_size - (kernel_size / 2); ++index_j)
		{
			float sum = 0.0;
			for (int i = -kernel_size / 2; i <= kernel_size / 2; ++i)
			{
				float data = image[index_i + i][index_j];
				float coeff = kernel[i + (kernel_size / 2)];

				sum += data * coeff;

				//cout << data << "\t" << coeff << "\t" << sum << endl;

				if (sum < min_value)
					min_value = sum;
				if (sum > max_value)
					max_value = sum;
			}
			output_image[index_i - kernel_size / 2][index_j - kernel_size / 2] = sum;
			//cout << sum << endl;
			//cout << output_image[index_i - kernel_size / 2][index_j - kernel_size / 2] << endl;
		}
	}
	//cout << min_value << endl << max_value << endl;

	for (int index_i = kernel_size / 2; index_i < y_size - (kernel_size / 2); ++index_i)
	{
		for (int index_j = kernel_size / 2; index_j < x_size - (kernel_size / 2); ++index_j)
		{
			int value = output_image[index_i - kernel_size / 2][index_j - kernel_size / 2];
			output_image[index_i - kernel_size / 2][index_j - kernel_size / 2] = 255 * (value - min_value) / (max_value - min_value);

		}
	}
}


int main(int argc, char** argv)
{
	const char* file = NULL;
	float s = 0;

	file = argv[1];
	s = atoi(argv[2]);

	cout << "filename: " << file << endl;
	cout << "sigma: " << s << endl << endl;

	//const float sigma = 1.0;
	const float sigma = s;
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
	/*
	for (int i = 0; i < 128; i++)
	{
		cout << rect_128[i] <<  " ";
	}
	cout << endl << endl;
	*/

	//Mat orig_signal = Mat(1, 128, CV_32F, rect_128);
	//Mat orig_signal_resized;
	//resize(orig_signal, orig_signal_resized, cv::Size(640, 240), 0, 0, cv::INTER_AREA);
	//imshow("Rect_128 (Original)", orig_signal_resized);
	//imwrite("Rect_128 (Original).jpg", orig_signal_resized);

	// Pad signal with zeros
	float rect_128_padded[padded_size];
	padd_with_zeros_1D(rect_128, rect_128_padded, 128, 5);
	/*
	for (int i = 0; i < 128; i++)
	{
		cout << rect_128_padded[i] << " ";
	}
	cout << endl << endl;
	*/

	// Create Gaussian mask 
	float Gaussian_Kernel_1D[mask_size];
	generate_gaussian_kernel_1D(Gaussian_Kernel_1D, sigma, mask_size);
	/*
	for (int i = 0; i < mask_size; i++)
	{
		cout << Gaussian_Kernel_1D[i] << " ";
	}
	cout << endl << endl;
	*/

	// Apply Gaussian smoothing to signal
	float rect_128_smoothed[128];
	apply_gaussian_smoothing_1D(rect_128_padded, padded_size, Gaussian_Kernel_1D, mask_size, rect_128_smoothed);
	/*
	for (int i = 0; i < 128; i++)
	{
		cout << rect_128_smoothed[i] << " ";
	}
	cout << endl << endl;
	*/

	// Write smoothed Rect_128 to .txt file
	ofstream ofs("Rect_128_smoothed.txt");
	for (int i = 0; i < 128; i++)
	{
		float f = rect_128_smoothed[i];
		ofs << f;
		ofs << "\n";
	}

	// Display smoothed signal
	//Mat smoothed_signal = Mat(1, 128, CV_32F, rect_128_smoothed);
	//Mat smoothed_signal_resized;
	//resize(smoothed_signal, smoothed_signal_resized, cv::Size(640, 240), 0, 0, cv::INTER_AREA);
	//imshow("Rect_128 (Smoothed)", smoothed_signal_resized);
	//imwrite("Rect_128 (Smoothed).jpg", smoothed_signal_resized);
	
	cout << "Rect_128 signal saved as Rect_128_smoothed.txt" << endl;
	

	//////////////////////////////////////////////////////
	/* 2D Gaussian Smooting */
	// 2D Kernel Smoothing
	int** input, ** output;
	int x_size, y_size, Q;

	//char name[20];
	//strncpy(name, file, sizeof(name) - 1);
	char name[20] = "lenna.pgm";
	
	char outfile_smoothed_2D[25] = "smoothed_2D.pgm";
	char outfile_smoothed_1D[25] = "smoothed_1D.pgm";

	ReadImage(name, &input, x_size, y_size, Q);
	
	// Generate 2D Gaussian kernel
	float** Gaussian_Kernel_2D;
	Gaussian_Kernel_2D = new float* [mask_size];
	for (int i = 0; i < mask_size; i++)
		Gaussian_Kernel_2D[i] = new float[mask_size];
	generate_gaussian_kernel_2D(Gaussian_Kernel_2D, sigma, mask_size);

	// Pad image with zeros
	int** input_padded_2D;
	input_padded_2D = new int* [y_size + mask_size - 1];
	for (int i = 0; i < y_size + mask_size - 1; i++)
		input_padded_2D[i] = new int[x_size + mask_size - 1];
	padd_with_zeros_2D(input, input_padded_2D, x_size, y_size, mask_size);
	

	// Apply Gaussian smoothing to image
	int** input_smoothed_2D;
	input_smoothed_2D = new int* [y_size];
	for (int i = 0; i < y_size; i++)
		input_smoothed_2D[i] = new int[x_size];
	for (int i = 0; i < y_size; i++)
		for (int j = 0; j < x_size; j++)
			input_smoothed_2D[i][j] = 0;

	apply_gaussian_smoothing_2D(input_padded_2D, x_size + mask_size - 1, y_size + mask_size - 1, Gaussian_Kernel_2D, mask_size, input_smoothed_2D);
	//apply_gaussian_smoothing_2D(input_padded, x_size, y_size, Gaussian_Kernel_2D, mask_size, input_smoothed);
	
	WriteImage(outfile_smoothed_2D, input_smoothed_2D, x_size, y_size, Q);
	cout << name << " smoothed with 2D kernel saved as " << outfile_smoothed_2D << endl;


	// 2D Gaussian Smoothing with 1D Convolution
	ReadImage(name, &input, x_size, y_size, Q);

	// Generate 1D Gaussian kernel 
	float Gaussian_Kernel_2D_1D[mask_size];
	generate_gaussian_kernel_1D(Gaussian_Kernel_2D_1D, sigma, mask_size);
	
	// Pad image with zeros
	int** input_padded_2D_1D;
	input_padded_2D_1D = new int* [y_size + mask_size - 1];
	for (int i = 0; i < y_size + mask_size - 1; i++)
		input_padded_2D_1D[i] = new int[x_size + mask_size - 1];
	padd_with_zeros_2D(input, input_padded_2D_1D, x_size, y_size, mask_size);

	// Apply Gaussian smoothing to image
	int** input_smoothed_2D_1D;
	input_smoothed_2D_1D = new int* [y_size];
	for (int i = 0; i < y_size; i++)
		input_smoothed_2D_1D[i] = new int[x_size];

	for (int i = 0; i < y_size; i++)
		for (int j = 0; j < x_size; j++)
			input_smoothed_2D_1D[i][j] = 0;

	apply_gaussian_smoothing_2D_with_1D(input_padded_2D_1D, x_size, y_size, Gaussian_Kernel_2D_1D, mask_size, input_smoothed_2D_1D);


	WriteImage(outfile_smoothed_1D, input_smoothed_2D_1D, x_size, y_size, Q);
	cout << name << " smoothed with 1D kernel saved as " << outfile_smoothed_1D << endl;


	//waitKey(0);
	return 0;
}