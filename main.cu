/**
**************************************************************************
*/

#include "Common.h"

//#define SCALEOUTPUT
#define OUTPUT

//include Kernels

#include "laplacian_kernel.cu"

void WrapperCUDA_pixME(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{

	//allocate device memory
	float *Src;
	float *Dst;
	size_t DstStride;
        
/*Allocate memory in the Device  */

	cudaMallocPitch((void **)(&Src), &DstStride, Size.width * sizeof(float), Size.height);
        cudaMemset2D((void *)(Src), DstStride,0, Size.width * sizeof(float), Size.height);


	cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(float), Size.height);
	cudaMemset2D((void *)(Dst), DstStride,0, Size.width * sizeof(float), Size.height);


	DstStride /= sizeof(float);

	//convert source image to float representation

	int ImgSrcFStride;
	float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);	

	CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);

	//Copy from host memory to device
        cudaMemcpy2D(Src, ImgSrcFStride * sizeof(float),
        ImgSrcF, ImgSrcFStride * sizeof(float), Size.width * sizeof(float), Size.height,cudaMemcpyHostToDevice);


	//setup execution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);
        
	// TODO: the algorithm should be as follows: 
	// (1) find the edges with the laplacian kernel = dI
	// (2) baseline correct and scale the laplacian "edges" image to 0,255 =  dI'
	// (3) subtract the corrected edges from the image -- I' = I-dI'
	// (4) correct and scale image = I-sharpened
	//execute CUDA kernel 
	CUDAkernel_Laplacian<<< grid, threads >>>(Src,Dst, (int) DstStride, (int) Size.height);

//	CUDAkernel_getLimits<<< grid, threads >>>(Dst, (int) DstStride, (int*) max, (int*) min);
	
	
	// find the minimum intensity in the image	
	/* TODO: this would be a good place to do the scaling and baseline correction */
	

	//Copy image block to host
	cudaMemcpy2D(ImgSrcF, ImgSrcFStride * sizeof(float), 
	Dst, DstStride * sizeof(float),	Size.width * sizeof(float), Size.height,
			cudaMemcpyDeviceToHost);

	//Convert image back to byte representation
	CopyFloat2Byte(ImgSrcF, ImgSrcFStride, ImgDst, Stride, Size);

	cudaFree(Dst);
	cudaFree(Src);
	FreePlane(ImgSrcF);


}

/**************************************************************************
*  Program entry point
*/

int main(int argc, char** argv)
{
	//initialize CUDA


	//source and results image filenames
	char SampleImageFnameResCUDA1[] = "laplacian_frame1.bmp";

	char *pSampleImageFpath ="data/frame1.bmp";	
	
	
	//preload image (acquire dimensions)
	int ImgWidth, ImgHeight;
	ROI ImgSize;
	int res = PreLoadBmp(pSampleImageFpath, &ImgWidth, &ImgHeight);
	ImgSize.width = ImgWidth;
	ImgSize.height = ImgHeight;
#ifdef SCALEOUTPUT
	int i, j, index;
	byte min=0, max=0, range, pixel, value;
#endif

	//CONSOLE INFORMATION: saying hello to user
	printf("CUDA Image Sharpning \n");
	printf("===================================\n");
	printf("Loading test image: %s... ", pSampleImageFpath);
	
	if (res)
	{
		printf("\nError: Image file not found or invalid!\n");
		printf("Press ENTER to exit...\n");
		getchar();

		//finalize
		exit(0);
	}

	//check image dimensions are multiples of BLOCK_SIZE
	if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0)
	{
		printf("\nError: Input image dimensions must be multiples of 8!\n");
		printf("Press ENTER to exit...\n");
		getchar();

		//finalize
		exit(0);
	}
	printf("[%d x %d]... ", ImgWidth, ImgHeight);


/**********************************************************************/
	//allocate image buffers
	int ImgStride;
	byte *ImgSrc = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
	byte *ImgDstCUDA1 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);

	//load sample image
	LoadBmpAsGray(pSampleImageFpath, ImgStride, ImgSize, ImgSrc);	
	

/******	// RUNNING WRAPPERS************************************************/
        printf("Success\nRunning CUDA 1 (GPU) version... ");

	WrapperCUDA_pixME(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);

/*********************************************************************************/

#ifdef DEBUG
	printf("ImgWidth,%d, ImgHeight,%d\n", ImgWidth, ImgHeight);
#endif
#ifdef SCALEOUTPUT
	/* determine min and max of image - should be with a kernel */
	for(i = 0; i < ImgWidth - 1; i++){
		for (j = 0; j < ImgHeight - 1; j++) {

			pixel = ImgDstCUDA1[i*ImgWidth + j]; /* get pixel */

			if(pixel < min)
				min = pixel; 		
			else if(pixel > max)
				max = pixel; 		
		}
	}
	/* baseline correct and scale image - should be with a kernel*/
	range = max - min;
	for(i = 0; i < ImgWidth - 1; i++) {
		for (j = 0; j < ImgHeight - 1; j++) {
			index = i * ImgWidth + j;
#ifdef DEBUG
	printf("index,%d\n",index);
#endif
			pixel = ImgDstCUDA1[index];
			value =  (pixel - min) / (range) * 255;
#ifdef VERBOSE
			printf("old,%d, new,%d, range,%d\n",pixel,value,range);
#endif
			ImgDstCUDA1[index] = value;
		}
	}
#endif

#ifdef OUTPUT
	//dump result of CUDA 1 processing
	printf("Success\nDumping result to %s... ", SampleImageFnameResCUDA1);
	DumpBmpAsGray(SampleImageFnameResCUDA1, ImgDstCUDA1, ImgStride, ImgSize);
#endif /* output */

	//print speed info
	printf("Success\n");


	//release byte planes
	FreePlane(ImgSrc);

	//finalize

	return 0;
}
