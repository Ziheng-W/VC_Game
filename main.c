#include "stdio.h"
#include "stdint.h"
#include "stdlib.h"
#include "param_noise.h"
#include "param_xin.h"
#include "param_yuan.h"
#include "cnn_basic.h"
#include "math.h"
#include <string.h>
float comp1();
float comp2();
float comp3();

float input[20][100] = {0};

int main(){
  comp1();
  comp2();
  comp3();

  return 0;
}

float comp1(){
  for (int i=0; i<20; i++){
    for (int j=0; j<100; j++){
      input[i][j] = 1;
    }
  }


  float *conv_1 = calloc(20*32*100, sizeof(float));
  ConvLayer_20_100 (TYPE_FLOAT, 2, input, 1, layer1_Conv2d_weight_noise, 3, 3, layer1_Conv2d_bias_noise,  conv_1, 32);
  b_norm(conv_1, 2000, 32, layer2_BatchNorm2d_weight_noise, layer2_BatchNorm2d_bias_noise, layer2_BatchNorm2d_running_mean_noise, layer2_BatchNorm2d_running_var_noise);
  ReLu(TYPE_FLOAT, conv_1, 2000*32, conv_1);
  float *pool_1 = calloc(10*32*50, sizeof(float));
  for (int32_t i=0; i<32; i++){
    MaxPool     (TYPE_FLOAT, 2, conv_1 + i*20*100, 20, 100, pool_1 + i*10*50);
  }
  free(conv_1);

  float *conv_2 = calloc(10*64*50, sizeof(float));
  ConvLayer_10_50 (TYPE_FLOAT, 2, pool_1, 32, layer3_Conv2d_weight_noise, 3, 3, layer3_Conv2d_bias_noise, conv_2, 64);
  free(pool_1);
  b_norm(conv_2, 10*50, 64, layer4_BatchNorm2d_weight_noise, layer4_BatchNorm2d_bias_noise, layer4_BatchNorm2d_running_mean_noise, layer4_BatchNorm2d_running_var_noise);
  ReLu(TYPE_FLOAT, conv_2, 500*64, conv_2);
  float *pool_2 = calloc(5*64*25, sizeof(float));
  for (int32_t i=0; i<64; i++){
    MaxPool     (TYPE_FLOAT, 2, conv_2 + i*10*50, 10, 50, pool_2 + i*5*25);
  }  
  free(conv_2);

  float *conv_3 = calloc(5*128*25, sizeof(float));
  ConvLayer_5_25 (TYPE_FLOAT, 2, pool_2, 64, layer5_Conv2d_weight_noise, 3, 3, layer5_Conv2d_bias_noise, conv_3, 128);
  free(pool_2);
  b_norm(conv_3, 125, 128, layer6_BatchNorm2d_weight_noise, layer6_BatchNorm2d_bias_noise, layer6_BatchNorm2d_running_mean_noise, layer6_BatchNorm2d_running_var_noise);  
  ReLu(TYPE_FLOAT, conv_3, 125*128, conv_3);
  float *pool_3 = calloc(128, sizeof(float));
  for(int32_t i=0; i<128; i++){
    AdaptivePool(TYPE_FLOAT, conv_3 + 25*5*i, 125, pool_3 + i);
  }
  free(conv_3);

  float result[2] = {0};
  FullConnect(TYPE_FLOAT, pool_3, 128, layer7_Linear_weight_noise, result, 2);
  free(pool_3);
  AddMatrix(TYPE_FLOAT, result, layer7_Linear_bias_noise, 2, result);
    
  // printf("comp_noise:\n");
  // print_matrix(TYPE_FLOAT, result, 2, 1);
  // printf("\n");
  return 0;
}

float comp2(){

  for (int i=0; i<20; i++){
    for (int j=0; j<100; j++){
      input[i][j] = 1;
    }
  }

  float * conv_1 = calloc(20*32*100, sizeof(float));
  ConvLayer_20_100 (TYPE_FLOAT, 2, input, 1, layer1_Conv2d_weight_xin, 3, 3, layer1_Conv2d_bias_xin,  conv_1, 32);
  ReLu(TYPE_FLOAT, conv_1, 2000*32, conv_1);
  float * pool_1 = calloc(10*32*50, sizeof(float));
  for (int32_t i=0; i<32; i++){
    MaxPool     (TYPE_FLOAT, 2, conv_1 + i*20*100, 20, 100, pool_1 + i*10*50);
  }
  free(conv_1);

  float * conv_2 = calloc(10*64*50, sizeof(float));
  ConvLayer_10_50 (TYPE_FLOAT, 2, pool_1, 32, layer2_Conv2d_weight_xin, 3, 3, layer2_Conv2d_bias_xin, conv_2, 64);
  free(pool_1);
  ReLu(TYPE_FLOAT, conv_2, 500*64, conv_2);
  float * pool_2 = calloc(5*128*25, sizeof(float));
  for (int32_t i=0; i<64; i++){
    MaxPool     (TYPE_FLOAT, 2, conv_2 + i*10*50, 10, 50, pool_2 + i*5*25);
  }  
  free(conv_2);

  float * conv_3 = calloc(5*128*25, sizeof(float));
  ConvLayer_5_25 (TYPE_FLOAT, 2, pool_2, 64, layer3_Conv2d_weight_xin, 3, 3, layer3_Conv2d_bias_xin, conv_3, 128);
  free(pool_2);
  ReLu(TYPE_FLOAT, conv_3, 125*128, conv_3);
  float * pool_3 = calloc(128, sizeof(float));
  for(int32_t i=0; i<128; i++){
    AdaptivePool(TYPE_FLOAT, conv_3 + 25*5*i, 125, pool_3 + i);
  }
  free(conv_3);

  float result[64] = {0};
  FullConnect(TYPE_FLOAT, pool_3, 128, layer4_Linear_weight_xin, result, 64);
  free(pool_3);
  AddMatrix(TYPE_FLOAT, result, layer4_Linear_bias_xin, 64, result);
  
  normalize_vector(result, 64);  

  // printf("comp_xin:\n");
  // print_matrix(TYPE_FLOAT, result, 64, 1);
  // printf("\n");
  return 0;
}

float comp3(){
    for (int i=0; i<20; i++){
    for (int j=0; j<100; j++){
      input[i][j] = 1;
    }
  }

  float * conv_1 = calloc(20*32*100, sizeof(float));
  ConvLayer_20_100 (TYPE_FLOAT, 2, input, 1, layer1_Conv2d_weight_yuan, 3, 3, layer1_Conv2d_bias_yuan,  conv_1, 32);
  ReLu(TYPE_FLOAT, conv_1, 2000*32, conv_1);
  float * pool_1 = calloc(10*32*50, sizeof(float));
  for (int32_t i=0; i<32; i++){
    MaxPool     (TYPE_FLOAT, 2, conv_1 + i*20*100, 20, 100, pool_1 + i*10*50);
  }
  free(conv_1);

  float * conv_2 = calloc(10*64*50, sizeof(float));
  ConvLayer_10_50 (TYPE_FLOAT, 2, pool_1, 32, layer2_Conv2d_weight_yuan, 3, 3, layer2_Conv2d_bias_yuan, conv_2, 64);
  free(pool_1);
  ReLu(TYPE_FLOAT, conv_2, 500*64, conv_2);
  float * pool_2 = calloc(5*128*25, sizeof(float));
  for (int32_t i=0; i<64; i++){
    MaxPool     (TYPE_FLOAT, 2, conv_2 + i*10*50, 10, 50, pool_2 + i*5*25);
  }  
  free(conv_2);

  float * conv_3 = calloc(5*128*25, sizeof(float));
  ConvLayer_5_25 (TYPE_FLOAT, 2, pool_2, 64, layer3_Conv2d_weight_yuan, 3, 3, layer3_Conv2d_bias_yuan, conv_3, 128);
  free(pool_2);
  ReLu(TYPE_FLOAT, conv_3, 125*128, conv_3);
  float * pool_3 = calloc(128, sizeof(float));
  for(int32_t i=0; i<128; i++){
    AdaptivePool(TYPE_FLOAT, conv_3 + 25*5*i, 125, pool_3 + i);
  }
  free(conv_3);

  float result[64] = {0};
  FullConnect(TYPE_FLOAT, pool_3, 128, layer4_Linear_weight_yuan, result, 64);
  free(pool_3);
  AddMatrix(TYPE_FLOAT, result, layer4_Linear_bias_yuan, 64, result);
  
  normalize_vector(result, 64);  

  // printf("comp_yuan:\n");
  // print_matrix(TYPE_FLOAT, result, 64, 1);
  // printf("\n");
  return 0;
}