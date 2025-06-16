#ifndef CNN_BASIC_H
#define CNN_BASIC_H
#include "stdio.h"
#include "stdint.h"
#include "stdlib.h"
// #include <math.h>

float fast_sqrt(float x);

typedef enum {
  TYPE_INT64_T,  // signed integer 
  TYPE_INT32_T, 
  TYPE_INT16_T,
  TYPE_INT8_T,
  TYPE_UINT64_T, // un-signed integer 
  TYPE_UINT32_T, 
  TYPE_UINT16_T,
  TYPE_UINT8_T,
  TYPE_FLOAT,    // floating point
  TYPE_DOUBLE 
} DataType;

typedef enum {
  DONE, 
  ERROR_EVEN_SIZE_KERNEL,
  ERROR_MISMATCH_IO_SIZE
} ReturnState;

typedef enum {
  MAX, 
  AVERAGE
} PoolTheme;

void print_matrix(DataType datatype, void * matrix, int32_t H, int32_t W);

float sum_matrix(void * input_, int size);

void bias(void *input, int32_t i_size, int32_t oc, float* bias);

void b_norm(void *input, int32_t i_size, int32_t oc, float* w, float * b, float * m, float * v);

void normalize_vector (void *input, int32_t length);

// 卷积计算
// 默认
// 1. 补零
// 2. 输入输出尺寸相同
// 3. 卷积核长宽均为奇数
ReturnState conv ( DataType datatype, 
            void * input, int32_t input_H, int32_t input_W,
            void * kernel, int32_t kernel_H, int32_t kernel_W,
            float bias,
            void * output
          );
// 全连接
// 默认： weight行长度=input长度
ReturnState FullConnect (DataType datatype, 
                  void * input, int32_t input_length,
                  void * weight,
                  void * output, int32_t output_length);

// 池化
ReturnState MaxPool (DataType datatype, int32_t poolsize,
                void * input, int32_t input_H, int32_t input_W, 
                void * output);

ReturnState ReLu (DataType datatype, void * input, int32_t input_length, void * output);

ReturnState AdaptivePool (DataType datatype, void * input, int32_t input_length, void * output);

ReturnState BatchNorm (DataType datatype, void * input, int32_t input_length, void * batch_weight, void * output);

ReturnState AddMatrix (DataType datatype, void * input_1, void * input_2, int32_t input_length, void * output);

ReturnState ReluAdaptivePool(DataType datatype, void * input, int32_t input_length, void * output);

ReturnState ConvLayer_20_100 (DataType datatype, int32_t poolsize,
                       void *input_, int32_t i_channel,
                       void *kernel_, int32_t kernel_H, int32_t kernel_W,
                       void *bias_, 
                       void *output_, int32_t o_channel);

ReturnState ConvLayer_10_50 (DataType datatype, int32_t poolsize,
                       void *input_, int32_t i_channel,
                       void *kernel_, int32_t kernel_H, int32_t kernel_W,
                       void *bias_, 
                       void *output_, int32_t o_channel);

ReturnState ConvLayer_5_25 (DataType datatype, int32_t poolsize,
                       void *input_, int32_t i_channel,
                       void *kernel_, int32_t kernel_H, int32_t kernel_W,
                       void *bias_, 
                       void *output_, int32_t o_channel);

#endif