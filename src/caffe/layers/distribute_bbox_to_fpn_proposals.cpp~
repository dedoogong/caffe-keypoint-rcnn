#include <algorithm>
#include <cfloat>
#include <vector>
#include <stdio.h>

#include "caffe/layers/collect_and_distribute_fpn_rpn_proposals.hpp"
#include "caffe/layers/distribute_bbox_to_fpn_proposals.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe { 

template <typename Dtype>
void DistributeBboxToFpnProposalsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { 
  CollectAndDistributeFpnRpnProposalsParameter fpn_proposal_param = this->layer_param_.fpn_proposal_param();

  roi_min_level_=fpn_proposal_param.roi_min_level();
  roi_canonical_level_=fpn_proposal_param.roi_canonical_level();
  roi_canonical_scale_=fpn_proposal_param.roi_canonical_scale();
  rpn_max_level_= fpn_proposal_param.rpn_max_level();
  roi_max_level_= fpn_proposal_param.roi_max_level();
  rpn_post_nms_topN_= fpn_proposal_param.rpn_post_nms_topn();
  rpn_min_level_= fpn_proposal_param.rpn_min_level();
}

template <typename Dtype>
void DistributeBboxToFpnProposalsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape_rois;
  vector<int> top_shape_keypoint_rois_fpn_2; 
  vector<int> top_shape_keypoint_rois_fpn_3;
  vector<int> top_shape_keypoint_rois_fpn_4;
  vector<int> top_shape_keypoint_rois_fpn_5;
  vector<int> top_shape_roi_index;
  
  int num_roi_2=1;
  int num_roi_3=1;
  int num_roi_4=1;
  int num_roi_5=1;  
 
  top_shape_keypoint_rois_fpn_2.push_back(num_roi_2);
  top_shape_keypoint_rois_fpn_2.push_back(5);
  top_shape_keypoint_rois_fpn_3.push_back(num_roi_3);
  top_shape_keypoint_rois_fpn_3.push_back(5);
  top_shape_keypoint_rois_fpn_4.push_back(num_roi_4);
  top_shape_keypoint_rois_fpn_4.push_back(5);
  top_shape_keypoint_rois_fpn_5.push_back(num_roi_5);
  top_shape_keypoint_rois_fpn_5.push_back(5);
  top_shape_roi_index.push_back(num_roi_2+num_roi_3+num_roi_4+num_roi_5);
  top_shape_roi_index.push_back(1);
  top[0]->Reshape(top_shape_keypoint_rois_fpn_2);//(num_roi_2, x1, y1, x2, y2) RPN proposals for ROI level 2, "
  top[1]->Reshape(top_shape_keypoint_rois_fpn_3);//(num_roi_3, x1, y1, x2, y2) RPN proposals for ROI level 3, "
  top[2]->Reshape(top_shape_keypoint_rois_fpn_4);//(num_roi_4, x1, y1, x2, y2) RPN proposals for ROI level 4, "
  top[3]->Reshape(top_shape_keypoint_rois_fpn_5);//(num_roi_5, x1, y1, x2, y2) RPN proposals for ROI level 5, "
  top[4]->Reshape(top_shape_roi_index);//(num_roi_2+num_roi_3+num_roi_4+num_roi_5) Permutation on the concatenation of all
}
template <typename Dtype>
void DistributeBboxToFpnProposalsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  printf("==============================================DistributeBboxToFpnProposalsLayer start=======================================\n");
  const float *im_info =(const float *)bottom[0]->cpu_data();
  const float *bbox_nms =(const float *)bottom[1]->cpu_data();
  float* roi_out;
  float detection_box[1000];
  printf("input size :[%d %d], [%d %d]\n",bottom[0]->shape(0), bottom[0]->shape(1), bottom[1]->shape(0), bottom[1]->shape(1));  
  printf("\n"); 
  printf("im_scale:%.3f\n",im_info[2]); 
  int detection_count=bottom[1]->shape(0);
  /*
    Detection Boxes to FPN
    equivalent to python code
    rois = boxes.astype(np.float, copy=False) * im_scale
    levels = np.zeros((boxes.shape[0], 1), dtype=np.int)
    rois_blob = np.hstack((levels, rois))
    rois_blob.astype(np.float32, copy=False) #[level, x1, y1, x2, y2] <-R x 5 matrix of RoIs in the image pyramid with columns
  */

  for(int i=0;i<detection_count;i++){
      detection_box[5*i]=bbox_nms[5*i];
      detection_box[5*i+1]=bbox_nms[5*i+1]*im_info[2];
      detection_box[5*i+2]=bbox_nms[5*i+2]*im_info[2];
      detection_box[5*i+3]=bbox_nms[5*i+3]*im_info[2];
      detection_box[5*i+4]=bbox_nms[5*i+4]*im_info[2];
      printf("%.2f %.2f %.2f %.2f %.2f\n",detection_box[5*i],detection_box[5*i+1],detection_box[5*i+2],detection_box[5*i+3],detection_box[5*i+4]); 
  }

  detection_box[0]=0.0f;
  detection_box[1]=134.7495f;
  detection_box[2]=67.00486f;
  detection_box[3]=565.65936f;
  detection_box[4]=779.6861f;

  caffe2::ERArrXXf rois(detection_count, 5);   
  Eigen::Map<const caffe2::ERArrXXf> roi(detection_box,detection_count, 5);//roi_in.data<float>(), 
  rois.block(0, 0, detection_count, 5) = roi; 

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Distribute
  // equivalent to python code
  //   lvl_min = cfg.FPN.ROI_MIN_LEVEL
  //   lvl_max = cfg.FPN.ROI_MAX_LEVEL
  //   lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)
  int num_roi_lvls = roi_max_level_ - roi_min_level_ + 1; 
  const int lvl_min = roi_min_level_;
  const int lvl_max = roi_max_level_;
  int canon_scale = roi_canonical_scale_;
  const int canon_level = roi_canonical_level_;
  auto rois_block = rois.block(0, 1, rois.rows(), 4); 

  int fpn_count[4]={0,0,0,0};
  auto lvls = utils::MapRoIsToFpnLevels(rois_block,
                                        lvl_min, lvl_max,
                                        canon_scale, canon_level, fpn_count); //canon_scale
  // equivalent to python code
  //   outputs[0].reshape(rois.shape)
  //   outputs[0].data[...] = rois
  float* rois_out =(float*)(top[0]->mutable_cpu_data());// Output(0);
  //rois_out->Resize(rois.rows(), rois.cols());
  Eigen::Map<caffe2::ERArrXXf> rois_out_mat(rois_out, rois.rows(), rois.cols());//->template mutable_data<float>()
  rois_out_mat = rois; 
  // Create new roi blobs for each FPN level
  // (See: modeling.FPN.add_multilevel_roi_blobs which is similar but annoying
  // to generalize to support this particular case.)
  //
  // equivalent to python code
  //   rois_idx_order = np.empty((0, ))
  //   for (output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)))
  //       idx_lvl = np.where(lvls == lvl)[0]
  //       blob_roi_level = rois[idx_lvl, :]
  //       outputs[output_idx + 1].reshape(blob_roi_level.shape)
  //       outputs[output_idx + 1].data[...] = blob_roi_level
  //       rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
  //   rois_idx_restore = np.argsort(rois_idx_order)
  //   blob_utils.py_op_copy_blob(rois_idx_restore.astype(np.int32), outputs[-1])
  caffe2::EArrXi rois_idx_restore; 
  
  
  for (int i = 0, lvl = lvl_min; i < num_roi_lvls; i++, lvl++) {//num_roi_lvls
    caffe2::ERArrXXf blob_roi_level;
    caffe2::EArrXi idx_lvl; 
    utils::RowsWhereRoILevelEquals(rois, lvls, lvl, &blob_roi_level, &idx_lvl);

    if(blob_roi_level.rows() == 0){
        printf("%dth fpn is empty\n",i);
        const vector< int > empty_roi_out_shape{0, 0};
        top[i]->Reshape(empty_roi_out_shape);
    }
    else{
        roi_out =(float*)top[i]->mutable_cpu_data();
        const vector< int > roi_out_shape{blob_roi_level.rows(), blob_roi_level.cols()};
        //printf("%dth fpn is not empty, size: %d\n",i,blob_roi_level.rows()*blob_roi_level.cols());

        for(int j=0;j<blob_roi_level.rows()*blob_roi_level.cols();j++){
              roi_out[j]=blob_roi_level.data()[j];
        }
        top[i]->Reshape(roi_out_shape);
    // Append indices from idx_lvl to rois_idx_restore
    rois_idx_restore.conservativeResize(rois_idx_restore.size() + idx_lvl.size());
    rois_idx_restore.tail(idx_lvl.size()) = idx_lvl;
    }
  } 
  printf("\n");
  utils::ArgSort(rois_idx_restore);
  int* rois_idx_restore_out = (int*)top[4]->mutable_cpu_data();
  Eigen::Map<caffe2::EArrXi> rois_idx_restore_out_mat(
      rois_idx_restore_out,
      rois_idx_restore.size());
  rois_idx_restore_out_mat = rois_idx_restore;

  const vector< int > rois_idx_shape{rois_idx_restore.size(),1};
  top[4]->Reshape(rois_idx_shape);
  printf("top shape : [%d, %d] [%d, %d] [%d, %d] [%d, %d] [%d, %d] \n", top[0]->shape(0),top[0]->shape(1) , 
                                                                        top[1]->shape(0),top[1]->shape(1) , 
                                                                        top[2]->shape(0),top[2]->shape(1) ,
                                                                        top[3]->shape(0),top[3]->shape(1) ,
                                                                        top[4]->shape(0),top[4]->shape(1));
  for(int j=0;j<top[4]->shape(0);j++)
     printf("roi idx : %d\n", rois_idx_restore_out[j]);                                       
  printf("==============================================DistributeBboxToFpnProposalsLayer done==========================\n");
}

#ifdef CPU_ONLY
STUB_GPU(DistributeBboxToFpnProposalsLayer);
#endif

INSTANTIATE_CLASS(DistributeBboxToFpnProposalsLayer);
REGISTER_LAYER_CLASS(DistributeBboxToFpnProposals);
}

