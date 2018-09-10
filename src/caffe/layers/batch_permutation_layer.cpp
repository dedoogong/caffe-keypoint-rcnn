#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/batch_permutation_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void BatchPermutationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { 
 
}

template <typename Dtype>
void BatchPermutationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->shape());
}

template <typename Dtype>
void BatchPermutationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const float* X =(const float*)bottom[0]->cpu_data();//Input(0);
    const int* indices =(const int*)bottom[1]->cpu_data();//Input(0);
    printf("==============================================BatchPermutationLayer start=====================================\n");

    printf("X shape : %d, %d, %d, %d indices shape : %d, %d \n", bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3), 
                                                                 bottom[1]->shape(0), bottom[1]->shape(1));
    // 1000, 256, 7,7
    float* Y =(float*)(top[0]->mutable_cpu_data());// Output(0); 
    const int N = bottom[0]->shape(0);
    const int C = bottom[0]->shape(1);
    const int H = bottom[0]->shape(2);
    const int W = bottom[0]->shape(3);

    const float *src = X;
    float *dst = Y;

    for (int i = 0; i < N; i++) {
      int idx = indices[i];
      //if(idx>=1000 || idx<0)
      //    printf("out of index range(0,1000) : %d\n",idx);
      std::memcpy(dst + i * C * H * W, src + idx * C * H * W, sizeof(float) * C * H * W);
    }
    printf("Y shape : %d, %d, %d, %d \n", top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));
    printf("==============================================BatchPermutationLayer done======================================\n");
    // 1000, 256, 7,7

}


#ifdef CPU_ONLY
STUB_GPU(BatchPermutationLayer);
#endif

INSTANTIATE_CLASS(BatchPermutationLayer);
REGISTER_LAYER_CLASS(BatchPermutation);
}

