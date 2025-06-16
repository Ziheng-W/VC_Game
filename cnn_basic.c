#include "cnn_basic.h"


void print_matrix(DataType datatype, void * matrix, int32_t H, int32_t W){
  for(int32_t i=0; i<H; i++){
    for (int32_t j=0; j<W; j++){
      if (datatype == TYPE_INT32_T){
        printf("%5d ", ((int32_t*)matrix)[i*W + j]);
      } 
      else if(datatype == TYPE_FLOAT){
        printf("%.8f ", ((float*)matrix)[i*W + j]);
      } 
      else {
        printf("%5.4lf ", ((double*)matrix)[i*W + j]);
      }
    }
    printf("\n");
  }
}

float sum_matrix(void * input_, int size){
  float * input = (float *) input_;
  float ret = 0;
  for(int i=0; i<size; i++){
    ret += input[i];
  }
  return ret;
}

void bias(void *input, int32_t i_size, int32_t oc, float* bias){
  for(int32_t j=0; j<oc; j++) {
    for(int32_t i=0; i<i_size; i++){
      ((float*)input)[j*i_size + i] += ((float*)bias)[j];
    }
  }
}

float fast_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    float guess = x;
    for (int i = 0; i < 8; ++i) {
        guess = 0.5f * (guess + x / guess);
    }
    return guess;
}

void b_norm(void *input, int32_t i_size, int32_t oc, float* w, float * b, float * m, float * v){
  for(int32_t j=0; j<oc; j++) {
    for(int32_t i=0; i<i_size; i++){
      ((float*)input)[j*i_size + i] = ((float*)w)[j] * (((float*)input)[j*i_size + i] - ((float*)m)[j]) / ( fast_sqrt(0.00001 + ((float*)v)[j]))   + ((float*)b)[j];
    }
  }
}

void normalize_vector (void *input, int32_t length){
  float distance = 0;
  for(int32_t i=0; i<length; i++){
    distance += ((float*)input)[i] * ((float*)input)[i];
  }
  distance = fast_sqrt(distance);
  if (distance < 1e-8f) return;
  for(int32_t i=0; i<length; i++){
    ((float*)input)[i] = ((float*)input)[i] / distance;
  }
}

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
          ){
  // 若卷积核为偶数尺寸，返回并报错
  int32_t kernel_mid_W = kernel_W >> 1;          
  int32_t kernel_mid_H = kernel_H >> 1;          
  if(kernel_mid_W << 1 == kernel_W || kernel_mid_H << 1 == kernel_H){
    return ERROR_EVEN_SIZE_KERNEL;
  }
  // 计算
  if(datatype == TYPE_INT32_T) {
 
  }
  else if(datatype == TYPE_FLOAT) {
    for(int32_t ii = 0; ii < input_H; ii++){
      for(int32_t ij = 0; ij < input_W; ij++){
        float partial_sum = bias;
        int32_t kernel_i = 0;
        for(int32_t i = -kernel_mid_H; i <= kernel_mid_H; i++){
          for(int32_t j = -kernel_mid_W; j <= kernel_mid_W; j++){
            int32_t input_i = ii + i;
            int32_t input_j = ij + j;
            if(input_i>=0 && input_i<input_H && input_j>=0 && input_j<input_W){
              partial_sum += ((float*)input)[input_i*input_W + input_j] * ((float*)kernel)[kernel_i];
            }
            kernel_i++;
          } 
        }
        ((float*)output)[ii*input_W + ij] = partial_sum; 
      }
    }
  }
  return DONE;
}

// 全连接
// 默认： weight行长度=input长度
ReturnState FullConnect (DataType datatype, 
                  void * input, int32_t input_length,
                  void * weight,
                  void * output, int32_t output_length){
  ;
  if(datatype == TYPE_INT32_T){
  }
  else if(datatype == TYPE_FLOAT){
    for (int32_t oi=0; oi < output_length; oi++){
      float partial_sum = 0;
      int32_t weight_base_i = oi * input_length;
      for (int32_t ii=0; ii < input_length; ii++){
        partial_sum += ((float*)input)[ii] * ((float*)weight)[weight_base_i + ii];
      }
      ((float*)output)[oi] = partial_sum;
    }
  } 
  return DONE;
}

// 池化
ReturnState MaxPool (DataType datatype, int32_t poolsize,
                void * input, int32_t input_H, int32_t input_W, 
                void * output) {
  int32_t output_H = input_H / poolsize;
  int32_t output_W = input_W / poolsize;                
  if (datatype == TYPE_INT32_T) {
    // int32_t ii = 0;
    // for (int32_t oi = 0; oi < output_H; oi++) {
    //   int32_t ij = 0;
    //   for (int32_t oj = 0; oj < output_W; oj++){
    //     int32_t temp = 0;
    //     for (int32_t pi = 0; pi < poolsize; pi++) {
    //       for (int32_t pj = 0; pj < poolsize; pj++) {
    //         int32_t temp_ = ((int32_t *)input)[(ii+pi)*input_W + ij+pj];
    //         temp = temp > temp_ ? temp : temp_;
    //       }
    //     }
    //     ((int32_t*)output)[oi*output_W + oj] = temp;
    //     ij += poolsize;
    //   }
    //   ii += poolsize;
    // }
  }
  else if (datatype == TYPE_FLOAT) {
    int32_t ii = 0;
    for (int32_t oi = 0; oi < output_H; oi++) {
      int32_t ij = 0;
      for (int32_t oj = 0; oj < output_W; oj++){
        float temp = 0;
        for (int32_t pi = 0; pi < poolsize; pi++) {
          for (int32_t pj = 0; pj < poolsize; pj++) {
            float temp_ = ((float *)input)[(ii+pi)*input_W + ij+pj];
            temp = temp > temp_ ? temp : temp_;
          }
        }
        ((float*)output)[oi*output_W + oj] = temp;
        ij += poolsize;
      }
      ii += poolsize;
    }
  }
  return DONE;
}

ReturnState ReLu (DataType datatype, void * input, int32_t input_length, void * output) {
  if (datatype == TYPE_INT32_T){
    for(int32_t i=0; i<input_length; i++){
      int32_t temp = ((int32_t *)input)[i];
      ((int32_t *)output)[i] = temp >= 0 ? temp : 0;
    }
  }
  else if (datatype == TYPE_FLOAT){
    for(int32_t i=0; i<input_length; i++){
      float temp = ((float *)input)[i];
      ((float *)output)[i] = temp >= 0 ? temp : 0;
    }
  }
  return DONE;
} 

ReturnState AdaptivePool (DataType datatype, void * input, int32_t input_length, void * output){
  if (datatype == TYPE_FLOAT){
    float temp = 0;
    for (int32_t i=0; i<input_length; i++){
      temp += ((float*)input)[i];
    }
    ((float*)output)[0] = temp / (float) input_length;
  }
  else if (datatype == TYPE_INT32_T){
    int32_t temp = 0;
    for (int32_t i=0; i<input_length; i++){
      temp += ((int32_t*)input)[i];
    }
    ((int32_t*)output)[0] = temp / input_length;
  }
  return DONE;
}

ReturnState BatchNorm (DataType datatype, void * input, int32_t input_length, void * batch_weight, void * output) {
  if (datatype == TYPE_FLOAT){
    float gamma = ((float*)batch_weight)[0];
    float beta = ((float*)batch_weight)[1];
    for (int32_t i=0; i<input_length; i++){
      ((float *)output)[i] = gamma * ((float*)input)[i] + beta;
    }
  }
  else if (datatype == TYPE_INT32_T){ // todo
    float gamma = ((float*)batch_weight)[0];
    float beta = ((float*)batch_weight)[1];
    for (int32_t i=0; i<input_length; i++){
      ((float *)output)[i] = gamma * ((float*)input)[i] + beta;
    }
  }
  return DONE;
}

ReturnState AddMatrix (DataType datatype, void * input_1, void * input_2, int32_t input_length, void * output){
  if (datatype == TYPE_FLOAT){
    for (int32_t i=0; i< input_length; i++){
      ((float *)output)[i] = ((float *)input_1)[i] + ((float *)input_2)[i]; 
    }
  }
  else if (datatype == TYPE_INT32_T){
    for (int32_t i=0; i< input_length; i++){
      ((int32_t *)output)[i] = ((int32_t *)input_1)[i] + ((int32_t *)input_2)[i]; 
    }
  }
  return DONE;
}

ReturnState ReluAdaptivePool(DataType datatype, void * input, int32_t input_length, void * output){
  if (datatype == TYPE_FLOAT){
    float temp;
    for (int32_t i=0; i<input_length; i++){
      if(((float*)input)[i] > 0){
        temp += ((float*)input)[i];
      }
    }
    ((float*)output)[0] = temp / (float) input_length;
  }
  else if (datatype == TYPE_INT32_T){
    int32_t temp;
    for (int32_t i=0; i<input_length; i++){
      if(((int32_t*)input)[i] > 0){
        temp += ((int32_t*)input)[i];
      }    
    }
    ((int32_t*)output)[0] = temp / input_length;
  }
  return DONE;  
}

ReturnState ConvLayer_20_100 (DataType datatype, int32_t poolsize,
                       void *input_, int32_t i_channel,
                       void *kernel_, int32_t kernel_H, int32_t kernel_W,
                       void *bias_, 
                       void *output_, int32_t o_channel) {
  const int32_t input_H = 20;
  const int32_t input_W = 100;
  if(datatype == TYPE_FLOAT){
    float * input  = (float *)input_;
    float * kernel = (float *)kernel_;
    float * bias  = (float *)bias_;
    float * output = (float *)output_;
    int32_t kernel_step = kernel_H*kernel_W;
    int32_t output_step = (input_H*input_W);
    int32_t input_size = input_H*input_W;
    int32_t bias_cnt = 0;
    for (int32_t i=0; i < o_channel; i++)
    { // Output channel
        float scratch_base [20][100] = {0};
        float scratch_incr [20][100] = {0};
        input  = (float *)input_;
        for (int32_t j=0; j < i_channel; j++)
        { // Input channel
            conv      (datatype, input, input_H, input_W, kernel, kernel_H, kernel_W, ((float *)bias)[bias_cnt], scratch_incr);
            AddMatrix (datatype, scratch_base, scratch_incr, input_size, output);  
            input += input_size;
            kernel += kernel_step;
            bias_cnt ++;
        }  
        output += output_step;
    } 
  } 
  else if(datatype == TYPE_INT32_T){  
  }        
  return DONE;         
}

ReturnState ConvLayer_10_50 (DataType datatype, int32_t poolsize,
                       void *input_, int32_t i_channel,
                       void *kernel_, int32_t kernel_H, int32_t kernel_W,
                       void *bias_, 
                       void *output_, int32_t o_channel) {
  const int32_t input_H = 10;
  const int32_t input_W = 50;
  if(datatype == TYPE_FLOAT){
    float * input  = (float *)input_;
    float * kernel = (float *)kernel_;
    float * bias  = (float *)bias_;
    float * output = (float *)output_;
    int32_t kernel_step = kernel_H*kernel_W;
    int32_t output_step = (input_H*input_W);
    int32_t input_size = input_H*input_W;
    for (int32_t i=0; i < o_channel; i++)
    { // Output channel
        input  = (float *)input_;
        float scratch_incr [10][50]={0};
        for (int32_t j=0; j < i_channel; j++)
        { // Input channel
            conv      (datatype, input, input_H, input_W, kernel, kernel_H, kernel_W, j == 0 ? ((float*)bias)[i] : 0, scratch_incr);
            AddMatrix (datatype, output, scratch_incr, input_size, output);  
            input += input_size;
            kernel += kernel_step;
        } 
        output += output_step;
    } 
  } 
  else if(datatype == TYPE_INT32_T){  
  }        
  return DONE;         
}

ReturnState ConvLayer_5_25 (DataType datatype, int32_t poolsize,
                       void *input_, int32_t i_channel,
                       void *kernel_, int32_t kernel_H, int32_t kernel_W,
                       void *bias_, 
                       void *output_, int32_t o_channel) {
  const int32_t input_H = 5;
  const int32_t input_W = 25;
  if(datatype == TYPE_FLOAT){
    float * input  = (float *)input_;
    float * kernel = (float *)kernel_;
    float * bias  = (float *)bias_;
    float * output = (float *)output_;
    int32_t kernel_step = kernel_H*kernel_W;
    int32_t output_step = (input_H*input_W);
    int32_t input_size = input_H*input_W;
    for (int32_t i=0; i < o_channel; i++)
    { // Output channel
        input  = (float *)input_;
        float scratch_incr [5][25]={0};
        for (int32_t j=0; j < i_channel; j++)
        { // Input channel
            conv      (datatype, input, input_H, input_W, kernel, kernel_H, kernel_W, j == 0 ? ((float*)bias)[i] : 0, scratch_incr);
            AddMatrix (datatype, output, scratch_incr, input_size, output);  
            input += input_size;
            kernel += kernel_step;
        } 
        output += output_step;
    } 
  } 
  else if(datatype == TYPE_INT32_T){  
  }        
  return DONE;         
}

