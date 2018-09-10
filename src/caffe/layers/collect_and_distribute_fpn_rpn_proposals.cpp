#include <algorithm>
#include <cfloat>
#include <vector>
#include <stdio.h>

#include "caffe/layers/collect_and_distribute_fpn_rpn_proposals.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe { 


    namespace utils {

        // Compute the area of an array of boxes.
        caffe2::ERArrXXf BoxesArea(const caffe2::ERArrXXf& boxes) {
          // equivalent to python code
          //   w = (boxes[:, 2] - boxes[:, 0] + 1)
          //   h = (boxes[:, 3] - boxes[:, 1] + 1)
          //   areas = w * h
          //   assert np.all(areas >= 0), 'Negative areas founds'
          const auto w = boxes.col(2) - boxes.col(0) + 1;
          const auto h = boxes.col(3) - boxes.col(1) + 1;
          const caffe2::ERArrXXf areas = w * h;
          /*
          int count=0;
          printf("=========================area under 100=========================\n");        
          for(int i=0;i<areas.rows();i++)
            if(areas.data()[i]<10.0f)
              count++;
          printf("count : %d",count);
              //printf("[x1 y1 x2 y2 = %.2f %.2f %.2f %.2f] %d'th area : %.5f\n",i, boxes.col(2)[i], boxes.col(0)[i], boxes.col(3)[i], boxes.col(1)[i], areas.data()[i]);

          count=0;  
          printf("\n=========================100 =< area < 10000=========================\n");        
          for(int i=0;i<areas.rows();i++)
            if(areas.data()[i]<10000.0f && areas.data()[i]>=10.0f)
              count++;
            //printf("[x1 y1 x2 y2 = %.2f %.2f %.2f %.2f] %d'th area : %.5f\n",i, boxes.col(2)[i], boxes.col(0)[i], boxes.col(3)[i], boxes.col(1)[i], areas.data()[i]);
          printf("count : %d",count); 

          count=0;
          printf("\n=========================10000 =< area < 30000=========================\n");        
          for(int i=0;i<areas.rows();i++)
            if(areas.data()[i]<40000.0f && areas.data()[i]>=10000.0f)
              count++;
            //printf("[x1 y1 x2 y2 = %.2f %.2f %.2f %.2f] %d'th area : %.5f\n",i, boxes.col(2)[i], boxes.col(0)[i], boxes.col(3)[i], boxes.col(1)[i], areas.data()[i]);
          printf("count : %d",count);

          count=0;
          printf("\n=========================30000 < area =========================\n");        
          for(int i=0;i<areas.rows();i++)
            if(areas.data()[i]>=40000.0f)
              count++;
            //printf("[x1 y1 x2 y2 = %.2f %.2f %.2f %.2f] %d'th area : %.5f\n",i, boxes.col(2)[i], boxes.col(0)[i], boxes.col(3)[i], boxes.col(1)[i], areas.data()[i]);
          printf("count : %d",count);
          */
          return areas;
        }

        // Determine which FPN level each RoI in a set of RoIs should map to based
        // on the heuristic in the FPN paper.
        caffe2::ERArrXXf MapRoIsToFpnLevels(Eigen::Ref<const caffe2::ERArrXXf> rois,
                                    const float k_min, const float k_max,//2,5
                                    const float s0, const float lvl0, int* fpn_count) {//4,224
          // Compute level ids
          caffe2::ERArrXXf s = BoxesArea(rois).sqrt();
          // s0 = cfg.FPN.ROI_CANONICAL_SCALE  # default: 224
          // lvl0 = cfg.FPN.ROI_CANONICAL_LEVEL  # default: 4

          // Eqn.(1) in FPN paper
          // equivalent to python code
          //   target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
          //   target_lvls = np.clip(target_lvls, k_min, k_max)
          auto target_lvls = (lvl0 + (s / s0 + 1e-6).log() / log(2)).floor();
          auto target_lvls_clipped = target_lvls.min(k_max).max(k_min);

          for(int i=0;i<target_lvls_clipped.rows();i++){
          
            if(target_lvls_clipped(i)==2.0)
              fpn_count[0]++;
            else if(target_lvls_clipped(i)==3.0)
              fpn_count[1]++;
            else if(target_lvls_clipped(i)==4.0)
              fpn_count[2]++;
            else if(target_lvls_clipped(i)==5.0)
              fpn_count[3]++;
          }

          printf("\ntarget_lvls size = %d\n",target_lvls.rows());
          printf("\ntarget_lvls_clipped size = %d\n",target_lvls_clipped.rows());
          printf("==================SCALE = %f=================\n",s0);
          printf("LEVEL 2= %d\n",fpn_count[0]);
          printf("LEVEL 3= %d\n",fpn_count[1]);
          printf("LEVEL 4= %d\n",fpn_count[2]);
          printf("LEVEL 5= %d\n",fpn_count[3]);

          return target_lvls_clipped;
        }

        // Sort RoIs from highest to lowest individual RoI score based on
        // values from scores array and limit to n results
        void SortAndLimitRoIsByScores(Eigen::Ref<const caffe2::EArrXf> scores, int n,
                                      caffe2::ERArrXXf& rois) {
          
          //CAFFE_ENFORCE(rois.rows() == scores.size(), "RoIs and scores count mismatch");
          // Create index array with 0, 1, ... N
          std::vector<int> idxs(rois.rows());
          std::iota(idxs.begin(), idxs.end(), 0);
          // Reuse a comparator based on scores and store a copy of RoIs that
          // will be truncated and manipulated below
          auto comp = [&scores](int lhs, int rhs) {
            if (scores(lhs) > scores(rhs)) return true;
            if (scores(lhs) < scores(rhs)) return false;
            // To ensure the sort is stable
            return lhs < rhs;
          };
          caffe2::ERArrXXf rois_copy = rois;
          // Note that people have found nth_element + sort to be much faster
          // than partial_sort so we use it here
          if (n > 0 && n < rois.rows()) {
            std::nth_element(idxs.begin(), idxs.begin() + n, idxs.end(), comp);
            rois.resize(n, rois.cols());
          } else {
            n = rois.rows();
          }
          std::sort(idxs.begin(), idxs.begin() + n, comp);
          // Update RoIs based on new order
          for (int i = 0; i < n; i++) {
            rois.row(i) = rois_copy.row(idxs[i]);
          }
        }

        // Updates arr to be indices that would sort the array. Implementation of
        // https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
        void ArgSort(caffe2::EArrXi& arr) {
          // Create index array with 0, 1, ... N and sort based on array values
          std::vector<int> idxs(arr.size());
          std::iota(std::begin(idxs), std::end(idxs), 0);
          std::sort(idxs.begin(), idxs.end(), [&arr](int lhs, int rhs) {
            return arr(lhs) < arr(rhs);
          });
          // Update array to match new order
          for (int i = 0; i < arr.size(); i++) {
            arr(i) = idxs[i];
          }
        }

        // Update out_filtered and out_indices with rows from rois where lvl matches
        // value in lvls passed in.
        void RowsWhereRoILevelEquals(Eigen::Ref<const caffe2::ERArrXXf> rois,
                                     const caffe2::ERArrXXf& lvls, const int lvl,
                                     caffe2::ERArrXXf* out_filtered, caffe2::EArrXi* out_indices) {
          //CAFFE_ENFORCE(out_filtered != nullptr, "Output filtered required");
          //CAFFE_ENFORCE(out_indices != nullptr, "Output indices required");
          //CAFFE_ENFORCE(rois.rows() == lvls.rows(), "RoIs and lvls count mismatch");
          // Calculate how many rows we need
          int filtered_size = (lvls == lvl).rowwise().any().count();
          // Fill in the rows and indices
          out_filtered->resize(filtered_size, rois.cols());
          out_indices->resize(filtered_size);
          for (int i = 0, filtered_idx = 0; i < rois.rows(); i++) {
            auto lvl_row = lvls.row(i);
            if ((lvl_row == lvl).any()) {
              out_filtered->row(filtered_idx) = rois.row(i);
              (*out_indices)(filtered_idx) = i;
              filtered_idx++;
            }
          }
        }

    } // namespace utils



template <typename Dtype>
void CollectAndDistributeFpnRpnProposalsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
void CollectAndDistributeFpnRpnProposalsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape_rois;
  vector<int> top_shape_roi_fpn_2;
  vector<int> top_shape_roi_fpn_3;
  vector<int> top_shape_roi_fpn_4;
  vector<int> top_shape_roi_fpn_5;
  vector<int> top_shape_roi_index;
  
  int total_num_rois=1000;
  int num_roi_2=250;
  int num_roi_3=250;
  int num_roi_4=250;
  int num_roi_5=250;  

  top_shape_rois.push_back(total_num_rois);
  top_shape_rois.push_back(4);
  top_shape_roi_fpn_2.push_back(num_roi_2);
  top_shape_roi_fpn_2.push_back(4);
  top_shape_roi_fpn_3.push_back(num_roi_3);
  top_shape_roi_fpn_3.push_back(4);
  top_shape_roi_fpn_4.push_back(num_roi_4);
  top_shape_roi_fpn_4.push_back(4);
  top_shape_roi_fpn_5.push_back(num_roi_5);
  top_shape_roi_fpn_5.push_back(4);
  top_shape_roi_index.push_back(num_roi_2+num_roi_3+num_roi_4+num_roi_5);
  top_shape_roi_index.push_back(1);
  top[0]->Reshape(top_shape_rois);//(total_num_rois, x1, y1, x2, y2) Top proposals ( rpn_post_nms_topN total <= rpn_post_nms_topN )
  top[1]->Reshape(top_shape_roi_fpn_2);//(num_roi_2, x1, y1, x2, y2) RPN proposals for ROI level 2, "
  top[2]->Reshape(top_shape_roi_fpn_3);//(num_roi_3, x1, y1, x2, y2) RPN proposals for ROI level 3, "
  top[3]->Reshape(top_shape_roi_fpn_4);//(num_roi_4, x1, y1, x2, y2) RPN proposals for ROI level 4, "
  top[4]->Reshape(top_shape_roi_fpn_5);//(num_roi_5, x1, y1, x2, y2) RPN proposals for ROI level 5, "
  top[5]->Reshape(top_shape_roi_index);//(num_roi_2+num_roi_3+num_roi_4+num_roi_5) Permutation on the concatenation of all
}
template <typename Dtype>
void CollectAndDistributeFpnRpnProposalsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  printf("==============================================CollectAndDistributeFpnRpnProposalsLayer start=====================\n");

  printf("rpn_rois_fpn2 size :%d\n",bottom[0]->shape(0));
  printf("rpn_rois_fpn3 size :%d\n",bottom[1]->shape(0));
  printf("rpn_rois_fpn4 size :%d\n",bottom[2]->shape(0));
  printf("rpn_rois_fpn5 size :%d\n",bottom[3]->shape(0));
  printf("rpn_rois_fpn6 size :%d\n",bottom[4]->shape(0));

  printf("rpn_roi_probs_fpn2 size :%d\n",bottom[5]->shape(0));
  printf("rpn_roi_probs_fpn3 size :%d\n",bottom[6]->shape(0));
  printf("rpn_roi_probs_fpn4 size :%d\n",bottom[7]->shape(0));
  printf("rpn_roi_probs_fpn5 size :%d\n",bottom[8]->shape(0));
  printf("rpn_roi_probs_fpn6 size :%d\n",bottom[9]->shape(0));

  int num_rpn_lvls = rpn_max_level_ - rpn_min_level_ + 1;
  CHECK_EQ(bottom.size(), 2 * num_rpn_lvls); //InputSize() 
  int num_roi_lvls = roi_max_level_ - roi_min_level_ + 1;
  //CAFFE_ENFORCE_EQ(top.size(), num_roi_lvls + 2);//OutputSize()

  // Collect rois and scores in Eigen
  // rois are in [[batch_idx, x0, y0, x1, y2], ...] format
  // Combine predictions across all levels and retain the top scoring
  //
  // equivalent to python code
  //   roi_inputs = inputs[:num_rpn_lvls]
  //   score_inputs = inputs[num_rpn_lvls:]
  //   rois = np.concatenate([blob.data for blob in roi_inputs])
  //   scores = np.concatenate([blob.data for blob in score_inputs]).squeeze()
  int proposal_num = 0;

  for (int i = 0; i < num_rpn_lvls; i++) {
    //const Dtype* roi_in = bottom[i]->cpu_data();//auto&->Dtype*, const Dtype* , Input(i);
    proposal_num += bottom[i]->num();//roi_in.dim(0) => bottom[i]->num() 
  } 

  caffe2::ERArrXXf rois(proposal_num, 5);
  caffe2::EArrXf scores(proposal_num); 
  int len = 0;
  printf("num_rpn_lvls :%d\n",num_rpn_lvls);
  for (int i = 0; i < num_rpn_lvls; i++) {
    const float* roi_in =(const float*)bottom[i]->cpu_data();//Input(i);
    int n = bottom[i]->shape(0);//roi_in.dim(0); 
    Eigen::Map<const caffe2::ERArrXXf> roi(roi_in,n, 5);//roi_in.data<float>(), 
    
    rois.block(len, 0, n, 5) = roi; 
    const float* score_in =(const float*)(bottom[num_rpn_lvls + i]->cpu_data());// auto& => Dtype*,  Input(num_rpn_lvls + i);

    // No need to squeeze, since we are reshaping when converting to Eigen
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html
    Eigen::Map<const caffe2::EArrXf> score(score_in, n);//score_in.data<float>()
    scores.segment(len, n) = score; 
    len += n;
    
  } 
  // Grab only top rpn_post_nms_topN rois
  // equivalent to python code
  //   inds = np.argsort(-scores)[:rpn_post_nms_topN]
  //   rois = rois[inds, :]
  printf("\n===========================BEFORE rois.data========================\n");
  for(int m=0;m<40;){
    printf("%.5f\t",rois.data()[m]);
    if(++m%5 == 0)
        printf("\n");
  }
  printf("SortAndLimitRoIsByScores BEFORE-> rois.rows(): (%d) ", rois.rows());//proposal_num
  utils::SortAndLimitRoIsByScores(scores, rpn_post_nms_topN_, rois); //code above
  printf("SortAndLimitRoIsByScores AFTER-> rois.rows(): (%d) ", rois.rows());//1000

  printf("\n===========================AFTER rois.data========================\n");
  for(int m=0;m<40;){
    printf("%.5f\t",rois.data()[m]);
    if(++m%5 == 0)
        printf("\n");
  }
  /*
  for(int j=0 ; j < rois.rows()  ;  j++){
      if(rois.data()[j*5 + 3] >10 )
        CHECK_GT(rois.data()[j*5 + 3],rois.data()[j*5 + 1]);
      if(rois.data()[j*5 + 4] >10 )
        CHECK_GT(rois.data()[j*5 + 4],rois.data()[j*5 + 2]);
  }
  */
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // Distribute
  // equivalent to python code
  //   lvl_min = cfg.FPN.ROI_MIN_LEVEL
  //   lvl_max = cfg.FPN.ROI_MAX_LEVEL
  //   lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)
  
  const int lvl_min = roi_min_level_;
  const int lvl_max = roi_max_level_;
  int canon_scale = roi_canonical_scale_;
  const int canon_level = roi_canonical_level_;
  auto rois_block = rois.block(0, 1, rois.rows(), 4); 
  int fpn_count[4]={0,0,0,0};
  auto lvls = utils::MapRoIsToFpnLevels(rois_block,
                                        lvl_min, lvl_max,
                                        canon_scale, canon_level,fpn_count); //canon_scale
  /*
    for(;;)
    if(zero_level){
      lvls = utils::MapRoIsToFpnLevels(rois_block,lvl_min, lvl_max, canon_scale--, canon_level,zero_level); //canon_scale
    }
    else{
      break; 
    }
  */
  // equivalent to python code
  //   outputs[0].reshape(rois.shape)
  //   outputs[0].data[...] = rois
  float* rois_out =(float*)(top[0]->mutable_cpu_data());// Output(0);
  //rois_out->Resize(rois.rows(), rois.cols());
  printf("rois.rows(): %d, rois.cols(): %d\n",rois.rows(), rois.cols());//rois shape == (1000,5)
  
  //Eigen::Map<caffe2::ERArrXXf> rois_out_mat(rois_out, rois.rows(), rois.cols());//->template mutable_data<float>()
  //rois_out_mat = rois; 

  vector<int> top_shape_rpn_rois;
  top_shape_rpn_rois.push_back(rois.rows());
  top_shape_rpn_rois.push_back(4);
  top[0]->Reshape(top_shape_rpn_rois);
  
  for(int i=0, j=0;i<rois.rows()*rois.cols();){
      if(j%5!=0)
          rois_out[i++]=rois.data()[j++];
      else
          j++;
  }

  /*
  for(int i=0;i<rois.rows()*(rois.cols()-1);){
      printf("%.3f   ",rois_out[i]);
      if(++i%4 == 0)
          printf("\n");
  }

  for(int i=0; i<rois.rows()*rois.cols();){
      printf("%.3f   ",rois.data()[i]);
      if(++i%5 == 0)
          printf("\n");
  }
  */
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
    // Output blob_roi_level
    float* roi_out =(float*)( top[i + 1]->mutable_cpu_data());//Output(i + 1);
    //roi_out->Resize(blob_roi_level.rows(), blob_roi_level.cols());
    const vector< int > roi_out_shape{blob_roi_level.rows(), blob_roi_level.cols()};
    //printf("\nblob_roi_level.rows(): (%d) blob_roi_level.cols(): (%d)\n", (int)blob_roi_level.rows(), (int)blob_roi_level.cols());
    if(blob_roi_level.rows() == 0){
        printf("%d's fpn is empty\n",lvl);
        const vector< int > empty_roi_out_shape{0, blob_roi_level.cols()};
        top[i + 1]->Reshape(empty_roi_out_shape);
    }
    else{
        top[i + 1]->Reshape(roi_out_shape);
    
        //printf("\nFPN roi counts : (%d)\n", top[i + 1]->shape(0));
        //Eigen::Map<caffe2::ERArrXXf> roi_out_mat(
        //    roi_out,//->template mutable_data<float>()
        //    blob_roi_level.rows(),
        //    blob_roi_level.cols());
        //roi_out_mat = blob_roi_level;
        for(int j=0;j<blob_roi_level.rows()*blob_roi_level.cols();j++){
              roi_out[j]=blob_roi_level.data()[j];
        }
        // Append indices from idx_lvl to rois_idx_restore
        rois_idx_restore.conservativeResize(rois_idx_restore.size() + idx_lvl.size());
        rois_idx_restore.tail(idx_lvl.size()) = idx_lvl;
    }
  }
  utils::ArgSort(rois_idx_restore);
  int* rois_idx_restore_out = (int*)top[5]->mutable_cpu_data();//Output(OutputSize() - 1);
  //rois_idx_restore_out->Resize(rois_idx_restore.size());

  printf("rois_idx_restore.size() : %d\n",rois_idx_restore.size());
  vector<int> top_shape_roi_index;
  top_shape_roi_index.push_back(rois_idx_restore.size());
  top_shape_roi_index.push_back(1);
  top[5]->Reshape(top_shape_roi_index);

  for(int i=0;i<rois_idx_restore.size();i++)
      rois_idx_restore_out[i]=rois_idx_restore.data()[i];

  printf("==============================================CollectAndDistributeFpnRpnProposalsLayer done=======================================\n");

}


#ifdef CPU_ONLY
STUB_GPU(CollectAndDistributeFpnRpnProposalsLayer);
#endif

INSTANTIATE_CLASS(CollectAndDistributeFpnRpnProposalsLayer);
REGISTER_LAYER_CLASS(CollectAndDistributeFpnRpnProposals);
}

