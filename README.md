README: CUDA-Enabled Laplacian Image Sharpening
	This project involved utilizing the CUDA tool set to enhance the contrast of an
image using Laplacian edge detection.  The nature of image processing is typically
“embarrassingly parallel” such that each pixel may be processed independently of all
others.  This makes Graphics Processing Units and their highly parallel architecture
well suited to image processing.  This program is run in emulation mode so no actual 
speedup can be calculated.  


Compilation instructions; 

module load cuda
make
./laplacian
