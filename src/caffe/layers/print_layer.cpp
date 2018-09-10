#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/print_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void PrintLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { 

    PrintParameter print_param = this->layer_param_.print_param(); 

    relu_cut= (int)print_param.relu_cut();
}

template <typename Dtype>
void PrintLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape());//(total_num_rois, x1, y1, x2, y2) Top proposals ( rpn_post_nms_topN total <= rpn_post_nms_topN ) 
}

template <typename Dtype>
void PrintLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const float*bottom_data=(const float*)bottom[0]->cpu_data();
  //printf("X shape : %d, %d\n", bottom[0]->shape(0), bottom[0]->shape(1));
  printf("==================================================PrintLayer Forward start==================================================\n");
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  if(bottom[0]->shape().size()==4){
      unsigned int N=1;//bottom[0]->shape(0);
      unsigned int C=2;//bottom[0]->shape(1);
      unsigned int H=bottom[0]->shape(2);  
      unsigned int W=bottom[0]->shape(3);
      if(relu_cut==0){
          for (int n = 0; n < N; ++n){
            for (int c = 0; c < C; ++c){
              for (int h = 0; h < H; ++h){
                for (int w = 0; w < W ; ++w){
                      printf("%.2f ",bottom_data[n*C*H*W+c*H*W+h*W+w]);
                }
                printf("\n");
              }
              printf("\n\n");
            }
            printf("\n\n\n");
          }
      }else{
          for (int n = 0; n < N; ++n){
            for (int c = 0; c < C; ++c){
              for (int h = 0; h < H; ++h){
                for (int w = 0; w < W ; ++w){
                      if(bottom_data[n*C*H*W+c*H*W+h*W+w]<0)
                          printf("%.2f ",0.0f);
                      else
                          printf("%.2f ",bottom_data[n*C*H*W+c*H*W+h*W+w]);
                }
                printf("\n");
              }
              printf("\n\n");
            }
            printf("\n\n\n");
          }
      }
  }
  else{
      for (int i = 0; i < count; ++i) {
        printf("%.2f ",bottom_data[i]);
      }
      printf("\n");
  }
  printf("==================================================PrintLayer Forward end==================================================\n");
}

#ifdef CPU_ONLY
STUB_GPU(PrintLayer);
#endif
INSTANTIATE_CLASS(PrintLayer);
REGISTER_LAYER_CLASS(Print);
}

