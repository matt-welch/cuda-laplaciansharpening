/**
**************************************************************************
*/

#include "Common.h"



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
        

	//execute CUDA kernel 
	CUDAkernel_Laplacian<<< grid, threads >>>(Src,Dst, (int) DstStride);
	

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




/**
**************************************************************************
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



	//dump result of CUDA 1 processing
	printf("Success\nDumping result to %s... ", SampleImageFnameResCUDA1);
	DumpBmpAsGray(SampleImageFnameResCUDA1, ImgDstCUDA1, ImgStride, ImgSize);

	//print speed info
	printf("Success\n");


	//release byte planes
	FreePlane(ImgSrc);

	//finalize

	return 0;
}
