//
// Created by Artyom on 12/6/2024.
//

#include "TextureObject.h"

TextureObject::TextureObject(torch::Tensor &tensor) {
    if (!tensor.is_cuda()) {
        throw std::runtime_error("Tensor must be on CUDA device");
    }

    // Ensure that the tensor is contiguous
    if (!tensor.is_contiguous()) {
        throw std::runtime_error("Tensor must be contiguous");
    }
    // Ensure the tensor is of the correct type, assuming we want uchar4
    //if (tensor.scalar_type() != torch::kUInt8 || tensor.size(-1) != 4) {
    //    std::cout<<tensor.scalar_type()<<" "<<tensor.size(-1)<<std::endl;
    //    throw std::runtime_error("Tensor must be of type uchar4 (unsigned 8-bit 4-channel)");
    //}

    std::cout<<tensor.data_ptr()<<std::endl;


    // Get tensor dimensions
    int width = tensor.size(1);   // Width of the image
    int height = tensor.size(0);  // Height of the image
    size_t pitch = width * 4;     // Each row is width * 4 bytes (uchar4)
    this->img_ptr = reinterpret_cast<CUdeviceptr>(tensor.data_ptr());

    // Create a CUDA array with uchar4 format
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
    std::cout<<"cuda array: ";
    std::cout<<cuArray<<std::endl;
    // Copy tensor data to CUDA array
    CUDA_CHECK(cudaMemcpy2DToArray(
            cuArray,
            0, 0,                                    // Destination offset
            tensor.data_ptr(),                       // Source pointer
            pitch,                                   // Pitch (row size in bytes)
            pitch,                                   // Width in bytes
            height,                                  // Height in rows
            cudaMemcpyDeviceToDevice                 // Tensor is already on the GPU
    ));

    // Create a CUDA resource descriptor
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Create texture descriptor
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    static cudaTextureObject_t last;
    CUDA_CHECK(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
    while (last == texture){
        CUDA_CHECK(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
    }
    last = texture;

    std::cout << "Tensor data pointer: " << tensor.data_ptr() << std::endl;
    std::cout<<"new image id: "<< texture <<std::endl;

}
TextureObject::~TextureObject()  {
    if (texture) {
        cudaDestroyTextureObject(texture);
    }
}