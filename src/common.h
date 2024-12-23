#ifndef COMMON_H
#define COMMON_H

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include "optixDynamicGeometry.h"
#include "../cuda/helpers.h"
#include "../cuda/random.h"
#include "../sutil/vec_math.h"
#include <curand.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>


// Ray type definitions
#define RAY_TYPE_RADIANCE 0
#define RAY_TYPE_COUNT 1  // Total number of ray types in the pipelin

#endif // COMMON_H
