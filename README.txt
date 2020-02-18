How to compile and run PA1_pb1

Using Command Line:
REQUIRED: G++ complier from GNU Compiler Collection

1. Unzip the "PA1_pb1.zip" folder and extract to a desired directory 
	
	e.g. C:\Users\Lee\source\repos\PA1_pb1

2. In Command Line, navigate to the main directory of the solution 
	
	e.g. cd C:\Users\Lee\source\repos\PA1_pb1\PA1_pb1

3. Compile PA1_pb1.cpp using the g++ compiler 
	
	e.g. g++ PA1_pb1.cpp -o PA1_pb1

4. Once compiled, run the excutable specifying the desired PGM filename and sigma value (NOTE: mask_size = 5 * sigma)
	
	e.g. To generate the Gaussian smoothed images with a sigma value of 5:  PA1_pb1 lenna.pgm 5

5. Final images will appear as:
	Rect_128_smoothed.txt (Smoothed Rect_128 signal)
	smoothed_2D.pgm (Input image smoothed with 2D kernel)
	smoothed_1D.pgm (Input image smoothed with 1D kernel)


