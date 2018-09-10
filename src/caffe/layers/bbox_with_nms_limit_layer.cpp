#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/bbox_with_nms_limit_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {
 
namespace {

  template <class Derived, class Func>
  vector<int> filter_with_indices(
      const Eigen::ArrayBase<Derived>& array,
      const vector<int>& indices,
      const Func& func) {
    vector<int> ret;
    for (auto& cur : indices) {
      if (func(array[cur])) {
        ret.push_back(cur);
      }
    }
    return ret;
  }
}
template <typename Dtype>
void BoxWithNMSLimitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
  BoxWithNMSLimitParameter box_nms_param = this->layer_param_.box_nms_param(); 

  score_thresh_=box_nms_param.score_thresh();
  nms_thresh_=box_nms_param.nms_thresh();
  detections_per_im_=box_nms_param.detections_per_im();
  soft_nms_enabled_= box_nms_param.soft_nms_enabled();
  soft_nms_method_= box_nms_param.soft_nms_method_();
  soft_nms_sigma_= box_nms_param.soft_nms_sigma();
  soft_nms_min_score_thresh_= box_nms_param.soft_nms_min_score_thresh();
  rotated_= box_nms_param.rotated(); 
}

template <typename Dtype>
void BoxWithNMSLimitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int nms_max_count = 300;// tscores.dim(1);
  vector<int> out_scores_shape;
  vector<int> out_boxes_shape;
  vector<int> out_classes_shape;

  out_scores_shape.push_back(nms_max_count);
  out_scores_shape.push_back(1);
  top[0]->Reshape(out_scores_shape);

  out_boxes_shape.push_back(nms_max_count);
  out_boxes_shape.push_back(5);
  top[1]->Reshape(out_boxes_shape);

  out_classes_shape.push_back(nms_max_count);
  out_classes_shape.push_back(1);
  top[2]->Reshape(out_classes_shape);

}

template <typename Dtype>
void BoxWithNMSLimitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const float* tscores =(const float*)bottom[0]->cpu_data();//cls_prob  blob (1000,2)
  const float* tboxes =(const float*)bottom[1]->cpu_data(); //pred_box  blob (1000,8) from bbox_transform layer
  float* out_scores = (float*)(top[0]->mutable_cpu_data()); //score_nms blob
  float* out_boxes =(float*)(top[1]->mutable_cpu_data());   //bbox_nms  blob
  float* out_classes = (float*)(top[2]->mutable_cpu_data());//class_nms blob

  //printf("Top shape : %d %d\n", top[0]->shape(0), top[0]->shape(1));
  printf("==============================================BoxWithNMSLimitLayer start====================================\n"); 
  /*
  for(int j=0;  j < bottom[1]->shape(0)*bottom[1]->shape(1)/100;  j++){// bg x1 y1 x2 y2, fg x1 y1 x2 y2
      if(j%8==0)
          printf("\n"); 
      printf("%.2f ",tboxes[j]);
  } 
  */
  //printf("cls_prob: [%d, %d]\n",bottom[0]->shape(0),bottom[0]->shape(1)); 
  //printf("pred_bbox: [%d, %d]\n",bottom[1]->shape(0),bottom[1]->shape(1)); 

  const int box_dim = 4;// rotated_ ? 5 :
  const int N = bottom[0]->shape(0); 
  // tscores: (num_boxes, num_classes), 0 for background
  
  CHECK_EQ(bottom[0]->shape().size(),2);//tscores.ndim(), 2
  CHECK_EQ(bottom[1]->shape().size(),2);//tboxes.ndim(),  2

  int num_classes = bottom[0]->shape(1);// tscores.dim(1);

  CHECK_EQ(N, bottom[1]->shape(0));//tboxes.dim(0));
  CHECK_EQ(num_classes * box_dim, bottom[1]->shape(1));// tboxes.dim(1));

  int batch_size = 1;
  vector<float> batch_splits_default(1, bottom[0]->shape(0));//tscores.dim(0)
  const float* batch_splits_data = batch_splits_default.data();

  Eigen::Map<const caffe2::EArrXf> batch_splits(batch_splits_data, batch_size);
  CHECK_EQ(batch_splits.sum(), N);
 
  //vector<int> total_keep_per_batch(batch_size);
  int offset = 0;
  int final_nms_count=0;
  for (int b = 0; b < batch_splits.size(); ++b) {// size == 1
      int num_boxes = batch_splits(b);// == 1000
      
      Eigen::Map<const caffe2::ERArrXXf> scores(
                                                tscores + offset * bottom[0]->shape(1),//tscores.dim(1),
                                                num_boxes,
                                                bottom[0]->shape(1));//tscores.dim(1));
      Eigen::Map<const caffe2::ERArrXXf> boxes(
                                                tboxes + offset * bottom[1]->shape(1),// tboxes.dim(1),
                                                num_boxes,
                                                bottom[1]->shape(1));//tboxes.dim(1));
 
  
      // To store updated scores if SoftNMS is used
      caffe2::ERArrXXf soft_nms_scores(num_boxes,  bottom[0]->shape(1));//tscores.dim(1));
      vector<vector<int>> keeps(num_classes);

      ///////////////////////// 1. Perform nms to each class /////////////////////////////////
      // skip j = 0, because it's the background class

      int total_keep_count = 0;

      for (int j = 1; j < num_classes; j++) {
        auto cur_scores = scores.col(j);

        auto inds = caffe2::utils::GetArrayIndices(cur_scores > score_thresh_);
        auto cur_boxes = boxes.block(0, j * box_dim, boxes.rows(), box_dim);

        if (soft_nms_enabled_) {
          auto cur_soft_nms_scores = soft_nms_scores.col(j);
          keeps[j] = caffe2::utils::soft_nms_cpu(
                                                  &cur_soft_nms_scores,
                                                  cur_boxes,
                                                  cur_scores,
                                                  inds,
                                                  soft_nms_sigma_,
                                                  nms_thresh_,
                                                  soft_nms_min_score_thresh_,
                                                  soft_nms_method_);
        } else {
          std::sort(
              inds.data(),
              inds.data() + inds.size(),
              [&cur_scores](int lhs, int rhs) {
                return cur_scores(lhs) > cur_scores(rhs);
              });
          keeps[j] = caffe2::utils::nms_cpu(cur_boxes, cur_scores, inds, nms_thresh_);
        }
        total_keep_count += keeps[j].size();
        //vector<int> cur_keeps=keeps[j];
        //for(int i=0; i<cur_keeps.size();i++)
        //    printf("cur_keeps[i] : %d\n",cur_keeps[i]);
        //printf("cur_keeps.size() : %d\n",cur_keeps.size());
      }
      printf("\ntotal_keep_count after step 1: %d\n",total_keep_count);//996
      if (soft_nms_enabled_) {
        // Re-map scores to the updated SoftNMS scores
        new (&scores) Eigen::Map<const caffe2::ERArrXXf>(
            soft_nms_scores.data(),
            soft_nms_scores.rows(),
            soft_nms_scores.cols());
      }

      /////////////////////// 2. Limit to max_per_image detections *over all classes* ///////////////////////////
      if (detections_per_im_ > 0 && total_keep_count > detections_per_im_) { // detections_per_im_ == 100
          // merge all scores together and sort
          auto get_all_scores_sorted = [&scores, &keeps, total_keep_count]() {
              caffe2::EArrXf ret(total_keep_count);

              int ret_idx = 0;
              
              for (int i = 1; i < keeps.size(); i++) {
                auto& cur_keep = keeps[i];
                auto cur_scores = scores.col(i);
                auto cur_ret = ret.segment(ret_idx, cur_keep.size());
                caffe2::utils::GetSubArray(cur_scores, caffe2::utils::AsEArrXt(keeps[i]), &cur_ret);
                ret_idx += cur_keep.size();
              }
              std::sort(ret.data(), ret.data() + ret.size());
              return ret;
          };
          // Compute image thres based on all classes
          auto all_scores_sorted = get_all_scores_sorted();
          CHECK_GT(all_scores_sorted.size(), detections_per_im_);
          auto image_thresh = all_scores_sorted[all_scores_sorted.size() - detections_per_im_];

          total_keep_count = 0;
              // filter results with image_thresh
              for (int j = 1; j < num_classes; j++) {
                  auto& cur_keep = keeps[j];
                  auto cur_scores = scores.col(j);
                  keeps[j] = filter_with_indices(
                      cur_scores, cur_keep, [&image_thresh](float sc) {
                          return sc >= image_thresh;
                      });
                  total_keep_count += keeps[j].size();
              }
      }
      printf("\ntotal_keep_count after step 2: %d\n",total_keep_count);//3
        
      //total_keep_per_batch[b] = total_keep_count;
      // Write results
      int cur_start_idx = top[0]->shape(0);
      int cur_out_idx = 0;
      float max_score=0.0f;
      int   max_idx=0;
      for (int j = 1; j < num_classes; j++) {
          auto  cur_scores = scores.col(j);
          auto  cur_boxes  = boxes.block(0, j * box_dim, boxes.rows(), box_dim);
          auto& cur_keep   = keeps[j]; // vector<vector<int>> keeps(num_classes);

          Eigen::Map<caffe2::EArrXf>  cur_out_scores( out_scores+ cur_start_idx + cur_out_idx                              , cur_keep.size());
          Eigen::Map<caffe2::ERArrXXf> cur_out_boxes( out_boxes + (cur_start_idx + cur_out_idx) * box_dim, cur_keep.size(), box_dim);
          Eigen::Map<caffe2::EArrXf> cur_out_classes( out_classes + cur_start_idx + cur_out_idx                            , cur_keep.size());

          caffe2::utils::GetSubArray(    cur_scores, caffe2::utils::AsEArrXt(cur_keep), &cur_out_scores);
          caffe2::utils::GetSubArrayRows( cur_boxes, caffe2::utils::AsEArrXt(cur_keep), &cur_out_boxes);
          if(0){
                printf("\n");
                for(int i=0; i<scores.rows();i++){
                   if(scores.col(1)[i]>0.6f)
                       printf("scores[%d] : %.2f box : [%.2f %.2f %.2f %.2f]\n",i,scores.col(1)[i], boxes.col(4)[i],boxes.col(5)[i],boxes.col(6)[i],boxes.col(7)[i]);
               
                } 
                for (int k = 0; k < cur_out_scores.size(); k++) {
                    if(cur_out_scores.data()[k]>max_score){
                        max_score=cur_out_scores.data()[k];
                        max_idx=k;
                    }     
                }
                
                printf("max_score :%.3f max_idx: %d\n",max_score, max_idx);
                for (int k = 0; k < cur_out_scores.size(); k++) {
                    out_scores[k]=cur_out_scores.data()[k];
                }
                for(int i=0; i<cur_boxes.rows();i++){
                    printf("cur_boxes[%d]: [%.2f %.2f %.2f %.2f]\n",i,cur_boxes.col(0)[i],cur_boxes.col(1)[i],cur_boxes.col(2)[i],cur_boxes.col(3)[i]);
                }       
                printf("max box : [%.2f %.2f %.2f %.2f] \n",out_boxes[4*max_idx],out_boxes[4*max_idx+1],out_boxes[4*max_idx+2],out_boxes[4*max_idx+3]);
          }
                    
          
          for(int i=0; i<cur_keep.size();i++){
                printf("cur_keep[%d]: %d\n",i,cur_keep[i]);
                printf("cur_scores[%d]: %.2f\n",i,cur_scores[cur_keep[i]]);
                out_boxes[5*i]  =b;
                out_boxes[5*i+1]=cur_boxes.col(0)[cur_keep[i]];
                out_boxes[5*i+2]=cur_boxes.col(1)[cur_keep[i]];
                out_boxes[5*i+3]=cur_boxes.col(2)[cur_keep[i]];
                out_boxes[5*i+4]=cur_boxes.col(3)[cur_keep[i]];
          } 
          max_score=0.0f;
          max_idx=0;
      }// end for (int j = 1; j < num_classes; j++) 
      offset += num_boxes;

      vector<int> out_scores_shape;
      vector<int> out_boxes_shape;
      vector<int> out_classes_shape;

      out_scores_shape.push_back(total_keep_count);
      out_scores_shape.push_back(1);
      top[0]->Reshape(out_scores_shape);

      out_boxes_shape.push_back(total_keep_count);
      out_boxes_shape.push_back(5);
      top[1]->Reshape(out_boxes_shape);

      out_classes_shape.push_back(total_keep_count);
      out_classes_shape.push_back(1);
      top[2]->Reshape(out_classes_shape);
      for(int i=0; i<total_keep_count;i++)
         printf("\nout_boxes: [%.2f, %.2f, %.2f, %.2f, %.2f]\n",out_boxes[5*i],out_boxes[5*i+1],out_boxes[5*i+2],out_boxes[5*i+3],out_boxes[5*i+4]); 
 
  }// end for (int b = 0; b < batch_splits.size(); ++b)

  printf("\n==============================================BoxWithNMSLimitLayer Done=====================================\n"); 
}

#ifdef CPU_ONLY
STUB_GPU(BoxWithNMSLimitLayer);
#endif

INSTANTIATE_CLASS(BoxWithNMSLimitLayer);
REGISTER_LAYER_CLASS(BoxWithNMSLimit);
}

