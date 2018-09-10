#ifndef CAFFE_ROI_ALIGN_LAYER_HPP_
#define CAFFE_ROI_ALIGN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <cstring>
namespace caffe {  
template <typename Dtype>
class RoIAlignLayer : public Layer<Dtype> {
 public:

  explicit RoIAlignLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top); 
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top); 

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  NOT_IMPLEMENTED;}
    float sampling_ratio;
		int pooled_w;
		int pooled_h;
		float spatial_scale;
};

}  // namespace caffe

#endif  // CAFFE_ROI_ALIGN_LAYER_HPP_
