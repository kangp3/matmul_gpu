#!/usr/bin/env python3

# %%
import torch
from torch.utils.cpp_extension import load_inline
from torchvision import io

# %%
img = io.read_image('octopus.jpg')
print(img.shape)

# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
# %%
%load_ext wurlitzer
# %%
cuda_begin = r'''
#include <torch/extensions.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }
'''
# %%
cuda_src = r'''
__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n) out[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n];
}

torch::Tensor rgb_to_grayscale(torch::Tensor input) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    printf("h*w: %d*%d\n", h, w);
    auto output = torch::empty({h, w}, input.options());
    int threads = 256;
    rgb_to_grayscale_kernel<<<cdiv(w*h, threads), threads>>>(
        input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
'''
# %%
cpp_src = r'''torch::Tensor rgb_to_grayscale(torch::Tensor input);'''
# %%
module = load_inline(
    cuda_sources=[cuda_src],
    cpp_sources=[cpp_src],
    functions=['rgb_to_grayscale'],
    name="inline_ext",
    verbose=True,
)
# %%
img_cuda = img.continuous().cuda()