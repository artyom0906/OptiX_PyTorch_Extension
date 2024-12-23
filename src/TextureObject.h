//
// Created by Artyom on 12/6/2024.
//

#ifndef OPTIX_PYTORCH_EXTENSION_TEXTUREOBJECT_H
#define OPTIX_PYTORCH_EXTENSION_TEXTUREOBJECT_H
#include <cuda.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "../sutil/Exception.h"

class TextureObject {
public:
    explicit TextureObject(torch::Tensor &tensor);

    ~TextureObject();

    cudaTextureObject_t get() {return texture;}
    CUdeviceptr get_img_ptr() {return img_ptr;}
private:
    cudaTextureObject_t texture = 0;
    CUdeviceptr img_ptr;
};


#endif//OPTIX_PYTORCH_EXTENSION_TEXTUREOBJECT_H
