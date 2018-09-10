#ifndef CAFFE_ResizeNearest_LAYER_HPP_
#define CAFFE_ResizeNearest_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <cstring>

namespace caffe {
 
template <typename Dtype>
class ResizeNearestLayer : public Layer<Dtype> {
 public:
  explicit ResizeNearestLayer(const LayerParameter& param)
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
  Dtype width_scale_;
  Dtype height_scale_;
};

}  // namespace caffe

#endif  // CAFFE_BATCH_PERMUTATION_LAYER_HPP_
