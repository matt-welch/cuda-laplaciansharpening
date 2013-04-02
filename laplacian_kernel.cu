
/**
**************************************************************************
*/
//#pragma once

#include "Common.h"
#define WORKING
#define COLLECTNEIGHBORS

__shared__ float CurBlockLocal1[BLOCK_SIZE2];
__shared__ float TargetBlockLocal2[BLOCK_SIZE2];


/* Read Pixel Value*/

__device__ float readPixVal( float* ImgSrc,int ImgWidth,int x,int y)
{
return (float)ImgSrc[y*ImgWidth+x];
}

__device__ void putPixVal( float* ImgDest,int ImgWidth,int x,int y, float pixVal)
{
	ImgDest[y*ImgWidth+x] = pixVal;
	return ;
}

__global__ void CUDAkernel_Laplacian(float *Src,float *Dst, int ImgWidth)
{
   	// Block index
 	const int bx = blockIdx.x ;
	const int by = blockIdx.y ;

   	// Thread index (current coefficient)
   	const int tx = threadIdx.x;
   	const int ty = threadIdx.y;

	// Texture coordinates
	const int tex_x=0 ;
	const int tex_y=0 ;

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

	//copy current image pixel to the first block
	// Read values into CurBlockLocal1 with readPixVal(Src,ImgWidth, tex_x, tex_y);
	int i,j;
//	printf("Tx, %d, ty, %d, BX, %d, BY, %d\n", threadIdx.x, threadIdx.y,
//			blockIdx.x, blockIdx.y);
	if(X > 0 && X < ImgWidth && Y > 0 && Y < ImgWidth){
#ifdef COLLECTNEIGHBORS
		for(i = 0; i < 3; ++i){
			for(j = 0; j < 3; ++j){
				CurBlockLocal1[j*3 + i] = readPixVal(Src,ImgWidth, X+i-1, Y+j-1);
			}
		}
#else
		CurBlockLocal1[4] = readPixVal(Src, ImgWidth, X, Y);
#endif

#ifdef WORKING
		/* perform the laplacian transform here TODO: incorrect calculation */
		newPixel =  CurBlockLocal1[0]*nborCoef + CurBlockLocal1[1]*nborCoef + CurBlockLocal1[2]*nborCoef + 
			CurBlockLocal1[3]*nborCoef + CurBlockLocal1[4]*centerCoef + CurBlockLocal1[5]*nborCoef + 
			CurBlockLocal1[6]*nborCoef + CurBlockLocal1[7]*nborCoef + CurBlockLocal1[8]*nborCoef; 
#else	
		newPixel = CurBlockLocal1[4];
#endif
	}
	else{
		newPixel = readPixVal(Src, ImgWidth, X, Y);
		state = "old";
	}
#ifdef DEBUG
	printf("x, %d, y, %d, %s, %f\n", X, Y, state, newPixel );
#endif
	//synchronize threads to make sure the block is copied
	__syncthreads();


	/*This is to copy the operated block to the destination image
	Use a similar function like readPixVal called putPixVal
	A smarter way is to copy all the block into the Dst array in parallel.. I will   leave that for you to discover.
	*/
	putPixVal(Dst, ImgWidth, X, Y, newPixel);
}


