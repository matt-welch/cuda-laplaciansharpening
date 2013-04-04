#/*******************************************************************************
# * FILENAME:    Makefile
# * DESCRIPTION: build options for CUDA Laplacian Sharpener (laplacian_kernel.cu) 
# * AUTHOR:      James Matthew Welch [JMW]
# * SCHOOL:      Arizona State University
# * CLASS:       CSE598: High Performance Computing
# * INSTRUCTOR:  Dr. Gil Speyer
# * SECTION:     20520
# * TERM:        Spring 2013
# *******************************************************************************/
#
all: laplacian

laplacian: laplacian_kernel.cu BmpUtil.cpp main.cu 
	nvcc BmpUtil.cpp main.cu -o laplacian -deviceemu

clean: 
	rm -rf laplacian *.o

debug: laplacian_kernel.cu BmpUtil.cpp main.cu
	nvcc BmpUtil.cpp main.cu -o laplacian -deviceemu -DDEBUG

tidy: clean
	rm -f *.*~ *~ laplacian_frame1.bmp

run: laplacian
	time ./laplacian
