#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_crop_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void PoolingCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { 
 
}

template <typename Dtype>
void PoolingCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->shape(0),bottom[0]->shape(1),bottom[0]->shape(2)-1,bottom[0]->shape(3)-1);
}

template <typename Dtype>
void PoolingCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const float* X =(const float*)bottom[0]->cpu_data();//Input(0); 
    printf("==============================================PoolingCropLayer start=======================================\n");
    //printf("X shape : %d, %d, %d, %d indices shape : %d, %d \n", bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3), 
    //                                                             bottom[1]->shape(0), bottom[1]->shape(1));
    // 1, 64, 201,2001
    
    float* Y =(float*)(top[0]->mutable_cpu_data());// Output(0); 
    const int N = bottom[0]->shape(0);
    const int C = bottom[0]->shape(1);
    const int H = bottom[0]->shape(2);
    const int W = bottom[0]->shape(3);

    const float *src = X;
    float *dst = Y;
    for (int n = 0; n < N; n++)
      for (int c = 0; c < C; c++)
        for (int h = 0; h < H-1; h++)
          std::memcpy(dst + n*C*(H-1)*(W-1) + c*(H-1)*(W-1) + h*(W-1), src + n*C*H*W + c*H*W + h*W, sizeof(float)*(W-1));

    printf("Y shape : %d, %d, %d, %d \n", top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));
    
    printf("==============================================PoolingCropLayer done=======================================\n");
    // 1000, 256, 7,7

}


#ifdef CPU_ONLY
STUB_GPU(PoolingCropLayer);
#endif

INSTANTIATE_CLASS(PoolingCropLayer);
REGISTER_LAYER_CLASS(PoolingCrop);
}

