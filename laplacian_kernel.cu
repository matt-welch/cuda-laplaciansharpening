/*******************************************************************************
 * FILENAME:    laplacian_kernel.cu 
 * DESCRIPTION: CUDA kernel performing Laplacian Image Sharpening 
 * AUTHOR:      James Matthew Welch [JMW]
 * SCHOOL:      Arizona State University
 * CLASS:       CSE598: High Performance Computing
 * INSTRUCTOR:  Dr. Gil Speyer
 * SECTION:     20520
 * TERM:        Spring 2013
 *******************************************************************************/
//#pragma once /* from original? */

#include "Common.h"
#define LAPLACIAN
#define COLLECTNEIGHBORS
#define DEBUG

__shared__ float CurBlockLocal1[BLOCK_SIZE2];
__shared__ float TargetBlockLocal2[BLOCK_SIZE2];


/* Read Pixel Value*/

__device__ float readPixVal( float* ImgSrc,int ImgWidth,int x,int y)
{
return (float)ImgSrc[y*ImgWidth+x];
}

/* put pixel value */
__device__ void putPixVal( float* ImgDest,int ImgWidth,int x,int y, float pixVal)
{
	ImgDest[y*ImgWidth+x] = pixVal;
	return ;
}

/* laplacian sharpening kernel */
__global__ void CUDAkernel_Laplacian(float *Src,float *Dst, int ImgWidth, int ImgHeight)
{
   	// Block index
 	const int bx = blockIdx.x ;
	const int by = blockIdx.y ;

   	// Thread index (current coefficient)
   	const int tx = threadIdx.x;
   	const int ty = threadIdx.y;

	// Texture coordinates /* unused but left in for reference */
//	const int tex_x=0 ;
//	const int tex_y=0 ;

	// my pixel coordinates
	const int X = bx * BLOCK_SIZE + tx;
	const int Y = by * BLOCK_SIZE + ty;

#ifdef DEBUG
	// messaging
	char* state = "new";
#endif

	// new pixel value
	float newPixel;
	const float centerCoef	= -7.0;
	const float nborCoef	= 1.0;
	const float coef[9] = {nborCoef, nborCoef, nborCoef, 
						nborCoef, centerCoef, nborCoef, 
						nborCoef, nborCoef, nborCoef};

	//copy current image pixel to the first block
	// Read values into CurBlockLocal1 with readPixVal(Src,ImgWidth, tex_x, tex_y);
	int i,j, sharpen=0;
	
#ifdef DEBUG
	//	printf("Tx, %d, ty, %d, BX, %d, BY, %d\n", threadIdx.x, threadIdx.y,
//			blockIdx.x, blockIdx.y);
#endif

	/* to save on branch testing: 
	 * create a logical variable representing all edges: 
	 * pixels without a zero coordinate & not at axis maxima */
	sharpen = (X!=0) & (Y!=0) & (X<(ImgWidth-1)) & (Y<(ImgHeight-1)); // & (ImgWidth-1-X) & (ImgHeight-1-Y);
#ifdef VERBOSE
	if (sharpen) {
		printf("sharpen,%d, x,%d, xdiff,%d, y,%d, ydiff,%d\n", sharpen,
			   	X, ImgWidth-X-1, Y, ImgHeight-Y-1);
	}
	//	printf("ROI size ( %d, %d)\n", ImgWidth, ImgHeight);
#endif
	if(sharpen){ /* pixels with non-zero testVal are not at edges (x!=0, xmax-x != 0*/
#ifdef COLLECTNEIGHBORS
		/* collect pixels in a 3x3 square */
		for(i = 0; i < 3; ++i){
			for(j = 0; j < 3; ++j){
				CurBlockLocal1[j*3 + i] = readPixVal(Src,ImgWidth, X+i-1, Y+j-1);
			}
		}
#else
		CurBlockLocal1[4] = readPixVal(Src, ImgWidth, X, Y);
#endif

#ifdef LAPLACIAN
		/* perform the laplacian sharpen here */
		newPixel =  
			CurBlockLocal1[0]*coef[0] + CurBlockLocal1[1]*coef[0]		+ CurBlockLocal1[2]*coef[2] + 
			CurBlockLocal1[3]*coef[3] + CurBlockLocal1[4]*centerCoef	+ CurBlockLocal1[5]*coef[5] + 
			CurBlockLocal1[6]*coef[6] + CurBlockLocal1[7]*coef[7]		+ CurBlockLocal1[8]*coef[8]; 
#else	
		newPixel = CurBlockLocal1[4];
#endif
	}else{ /* pixel on a border - no sharpening */
		newPixel = readPixVal(Src, ImgWidth, X, Y);
#ifdef DEBUG
		state = "old";
#endif
	}
#ifdef VERBOSE
	printf("x, %d, y, %d, %s, %3.2f\n", X, Y, state, newPixel);
#endif
	//synchronize threads to make sure the block is copied
	__syncthreads();


	/*This is to copy the operated block to the destination image
	Use a similar function like readPixVal called putPixVal
	A smarter way is to copy all the block into the Dst array in parallel.. I will   leave that for you to discover.
	*/
	putPixVal(Dst, ImgWidth, X, Y, newPixel);
}


