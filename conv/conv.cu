#include <stdio.h>
#include "CycleTimer.h"


__global__
void conv1dNoTiling(int n, int filterSize, float* input, float* output, float* filter){
    /*
    //each thread is responsible of doing one output element
    */

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n-filterSize+1){
        return;
    }

    output[idx]=0;
    for(int i=0;i<filterSize;i++){
        output[idx] += filter[i]*input[idx+i];
    }
}

__global__ 
void conv1dTiled(int n, int filterSize, float* input, float* output, float* filter){
    /*
    //a threadblock will fill a tw size of shared memory
    */

    int outputSize = n-filterSize+1;
    extern __shared__ float sh[];

    //local index
    int tx = threadIdx.x;
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx >= outputSize){
        return;
    }

    //debug: can't be out of bounds
    sh[tx] = input[idx];
    if(tx < filterSize-1 && idx + blockDim.x < n){
        sh[tx+blockDim.x] = input[idx+blockDim.x];
    }
    __syncthreads();

    float sum = 0;
    for(int i=0;i<filterSize;i++){
        sum += filter[i]*sh[tx+ i];
    }
    output[idx] = sum;
    __syncthreads();
}

int main(){
    // printf("hello from main\n");
    float* input;
    float* output;
    float* filter;

    int n = 1024*1024*10; //40 MB array
    int filterSize = 3;

    input = (float*)malloc(n*sizeof(float));
    output = (float*)malloc((n-filterSize+1)*sizeof(float));
    filter = (float*)malloc(filterSize*sizeof(float));

    /*
    //serial execution
    */
    for(int i=0; i<n; i++){
        input[i] = i;
    }

    for(int i=0; i<filterSize; i++){
        filter[i] = 1;
    }

    double startTime = CycleTimer::currentSeconds();
    for(int i=0; i<n-filterSize+1; i++){
        int sum=0;
        for(int j=0; j<filterSize; j++){
           sum += filter[j]*input[i+j];
        }
        output[i] = sum;
    }
    double endTime = CycleTimer::currentSeconds();

    printf("Time taken: %f, bandwidth [%fGB/s]\n", endTime-startTime, (n*sizeof(float)*3)/(endTime-startTime)/1024/1024/1024);


    /*
    //prepping for cuda execution
    */

    float* outputCuda;

    outputCuda = (float*)malloc((n-filterSize+1)*sizeof(float));

    float* inputc;
    float* outputc;
    float* filterc;

    

    cudaMalloc(&inputc, n*sizeof(float));
    cudaMalloc(&outputc, (n-filterSize+1)*sizeof(float));
    cudaMalloc(&filterc, filterSize*sizeof(float));



    cudaMemcpy(inputc, input, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(filterc, filter, filterSize*sizeof(float), cudaMemcpyHostToDevice);

    int tpb = 512;
    int totalThreads = n - filterSize + 1;
    int nb = (totalThreads+tpb-1)/tpb;

    startTime = CycleTimer::currentSeconds();
    conv1dNoTiling<<<nb,tpb>>> (n, filterSize, inputc, outputc, filterc);
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();

    printf("Time taken: %f, bandwidth [%fGB/s]\n", endTime-startTime, (n*sizeof(float)*3)/(endTime-startTime)/1024/1024/1024);
    
    cudaMemcpy(outputCuda, outputc, (n-filterSize+1)*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<n-filterSize-1; i++){
        if(outputCuda[i]!=output[i]){
            printf("output different at %d, %f and %f", i, output[i], outputCuda[i]);
            return 1;
        }
    }

    /*
    //prepping for tiled execution
    */

    int tw = tpb + filterSize - 1;
    size_t size = tw*sizeof(float);

    float* outputTile;
    float* outputTilec;

    outputTile = (float*)malloc((n-filterSize+1)*sizeof(float));
    cudaMalloc(&outputTilec, (n-filterSize+1)*sizeof(float));

    startTime = CycleTimer::currentSeconds();
    conv1dTiled<<<nb,tpb,size>>>(n, filterSize, inputc, outputTilec, filterc);
    cudaDeviceSynchronize();
    endTime = CycleTimer::currentSeconds();

    printf("Time taken: %f, bandwidth [%fGB/s]\n", endTime-startTime, (n*sizeof(float)*3)/(endTime-startTime)/1024/1024/1024);

    cudaMemcpy(outputTile, outputTilec, (n-filterSize+1)*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<n-filterSize-1; i++){
        if(outputTile[i]!=output[i]){
            printf("output different at %d, %f and %f", i, output[i], outputTile[i]);
            return 1;
        }
    }
    printf("\033[1;32mPassed!!!\033[0m\n");
    cudaFree(inputc);
    cudaFree(outputc);
    cudaFree(filterc);

    free(input);
    free(output);
    free(filter);
}