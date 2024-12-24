#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// CUDA kernel for upsweep phase
__global__ void upsweep_kernel(float* input, int two_d, int n) {
    int two_dplus1 = 2 * two_d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx % two_dplus1 == 0 && idx + two_dplus1 - 1 < n) {
        input[idx + two_dplus1 - 1] += input[idx + two_d - 1];
    }
}

// CUDA kernel for downsweep phase
__global__ void downsweep_kernel(float* input, int two_d, int n) {
    int two_dplus1 = 2 * two_d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx % two_dplus1 == 0 && idx + two_dplus1 - 1 < n) {
        float t = input[idx + two_d - 1];
        input[idx + two_d - 1] = input[idx + two_dplus1 - 1];
        input[idx + two_dplus1 - 1] += t;
    }
}

void prefix_sum(std::vector<float>& input) {

    float* d_input;
    size_t size = input.size() * sizeof(float);
    cudaMalloc(&d_input, size);
  
    cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);
 
    int tpb = 512;  
    int nb = (input.size() + tpb - 1) / tpb; 
    
    int two_d = 1;
    while (two_d <= input.size() / 2) {
        upsweep_kernel<<<nb, tpb>>>(d_input, two_d, input.size());
        two_d *= 2;
    }
    
    cudaMemset(d_input + input.size() - 1, 0, sizeof(float));
    
    two_d = input.size() / 2;
    while (two_d >= 1) {
        downsweep_kernel<<<nb, tpb>>>(d_input, two_d, input.size());
        two_d /= 2;
    }
    
    cudaMemcpy(input.data(), d_input, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
}

int main() {
    int n=64;
    std::vector<float> input(n);
    for (int i = 0; i < n/2+n/4+1; i++) {
        input[i] = i;
    }
    //place 1000 in others
    for (int i = n/2+n/4+1; i < n; i++) {
        input[i] = 1000;
    }
    
    //serial exclusive prefix sum
    std::vector<float> result = std::vector<float>(input.size());
    result[0] = 0;
    for (int i = 1; i < input.size(); i++) {
        result[i] = result[i - 1] + input[i - 1];
    }

    prefix_sum(input);
    
    std::cout << "Prefix Sum Result: ";
    for (float val : input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    //check correctness
    std::cout<<input.size()<<std::endl;
    for (int i = 0; i < input.size(); i++) {
        if (input[i] != result[i]) {
            std::cout << "Mismatch at index " << i << std::endl;
            return 1;
        }
    }
    
    return 0;
}