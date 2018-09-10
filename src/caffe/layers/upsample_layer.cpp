#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/upsample_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ResizeNearestLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  ResizeNearestParameter resize_nearest_param = this->layer_param_.resize_nearest_param();

  height_scale_=resize_nearest_param.height_scale();
  width_scale_ =resize_nearest_param.width_scale();
}

template <typename Dtype>
void ResizeNearestLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int batch_size = bottom[0]->shape(0);
  const int num_channels = bottom[0]->shape(1);
  const int input_height = bottom[0]->shape(2);
  const int input_width = bottom[0]->shape(3);

  int output_width  = input_width * width_scale_;
  int output_height = input_height * height_scale_;

  top[0]->Reshape(batch_size, num_channels, output_height, output_width);
}

template <typename Dtype>
void ResizeNearestLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const float* X =(const float*)bottom[0]->cpu_data();//Input(0);
  float* Y =(float*)(top[0]->mutable_cpu_data());// Output(0); 
  const int batch_size = bottom[0]->shape(0);
  const int num_channels = bottom[0]->shape(1);
  const int input_height = bottom[0]->shape(2);
  const int input_width = bottom[0]->shape(3);

  int output_width  = input_width * width_scale_;
  int output_height = input_height * height_scale_;

  //

  const float* input = X;
  float* output = Y;
  int channels = num_channels * batch_size;

  const float rheight = (output_height > 1) ? (float)(input_height - 1) / (output_height - 1) : 0.f;
  const float rwidth = (output_width > 1) ? (float)(input_width - 1) / (output_width - 1) : 0.f;
  for (int h2 = 0; h2 < output_height; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < input_height - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = (float)1. - h1lambda;
    for (int w2 = 0; w2 < output_width; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < input_width - 1) ? 1 : 0;
      const float w1lambda = w1r - w1;
      const float w0lambda = (float)1. - w1lambda;
      const float* Xdata = &input[h1 * input_width + w1];
      float* Ydata = &output[h2 * output_width + w2];
      for (int c = 0; c < channels; ++c) {
        Ydata[0] = h0lambda * (w0lambda * Xdata[0] + w1lambda * Xdata[w1p]) +
            h1lambda *
                (w0lambda * Xdata[h1p * input_width] +
                 w1lambda * Xdata[h1p * input_width + w1p]);
        Xdata += input_width * input_height;
        Ydata += output_width * output_height;
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ResizeNearestLayer);
#endif

INSTANTIATE_CLASS(ResizeNearestLayer);
REGISTER_LAYER_CLASS(ResizeNearest);
}

