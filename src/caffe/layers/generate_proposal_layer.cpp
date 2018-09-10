#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/generate_proposal_layer.hpp" 
#include "caffe/layers/generate_proposals_op_util_boxes.hpp"
using std::max;
using std::min;
using std::floor;
using std::ceil;

int limit = 40;// cur_bbox_deltas.size();

namespace caffe {
 
template <typename Dtype>
void GenerateProposalLayer<Dtype>:: ProposalsForOneImage(
    const Eigen::Array3f& im_info,
    const Eigen::Map<const caffe2::ERMatXf>& all_anchors,
    const utils::ConstTensorView<float>& bbox_deltas_tensor,
    const utils::ConstTensorView<float>& scores_tensor,
    caffe2::ERArrXXf* out_boxes,
    caffe2::EArrXf* out_probs) const {

  printf("==================================================ProposalsForOneImage==================================================\n");
  const auto& post_nms_topN = post_nms_topn_;// rpn_post_nms_topN_;
  const auto& nms_thresh = nms_thresh_;
  const auto& min_size = min_size_;
  const int box_dim = static_cast<int>(all_anchors.cols());
  CHECK_EQ(box_dim , 4)<< "box_dim should be dx dy dx dy";

  // Transpose and reshape predicted bbox transformations to get them
  // into the same order as the anchors:
  //   - bbox deltas will be (box_dim * A, H, W) format from conv output
  //   - transpose to (H, W, box_dim * A)
  //   - reshape to (H * W * A, box_dim) where rows are ordered by (H, W, A)
  //     in slowest to fastest order to match the enumerated anchors

  CHECK_EQ(bbox_deltas_tensor.ndim(), 3)<< "bbox_deltas_tensor should be 3D tensor";
  CHECK_EQ(bbox_deltas_tensor.dim(0) % box_dim, 0)<< "bbox_deltas_tensor should be multiplication of box_dim ";

  auto A = bbox_deltas_tensor.dim(0) / box_dim;
  auto H = bbox_deltas_tensor.dim(1);
  auto W = bbox_deltas_tensor.dim(2);
  // equivalent to python code
  //  bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape((-1, box_dim))
  caffe2::ERArrXXf bbox_deltas(H * W * A, box_dim);
  Eigen::Map<caffe2::ERMatXf>(bbox_deltas.data(), H * W, box_dim * A) =
      Eigen::Map<const caffe2::ERMatXf>(bbox_deltas_tensor.data(), A * box_dim, H * W)
          .transpose();
  CHECK_EQ(bbox_deltas.rows(), all_anchors.rows());

  if(DEBUG){
      printf("ProposalsForOneImage-[bbox_deltas]\n");
      for(int m=0;m<limit;){
        printf("%.5f\t",bbox_deltas.data()[m]);
        if(++m%5 == 0)
            printf("\n");
      }
  }
  // - scores are (A, H, W) format from conv output
  // - transpose to (H, W, A)
  // - reshape to (H * W * A, 1) where rows are ordered by (H, W, A)
  //   to match the order of anchors and bbox_deltas
  CHECK_EQ(scores_tensor.ndim(), 3);
  CHECK_EQ(scores_tensor.dims(0), (A));
  CHECK_EQ(scores_tensor.dims(1), (H));
  CHECK_EQ(scores_tensor.dims(2), (W));
  // equivalent to python code
  // scores = scores.transpose((1, 2, 0)).reshape((-1, 1))
  caffe2::EArrXf scores(scores_tensor.size());
  Eigen::Map<caffe2::ERMatXf>(scores.data(), H * W, A) =
      Eigen::Map<const caffe2::ERMatXf>(scores_tensor.data(), A, H * W).transpose();

  std::vector<int> order(scores.size()); // order is index for all_anchors_sorted and bbox_deltas_sorted
  std::iota(order.begin(), order.end(), 0);

  if(DEBUG){

      printf("ProposalsForOneImage-[scores]\n");
        for(int m=0;m<limit;){
          printf("%.5f\t",scores.data()[m]);
          if(++m%5 == 0)
              printf("\n");
        }
      
      printf("ProposalsForOneImage-[order] scores.size = %d\n",scores.size());
      for(int m=0;m<limit;){
          printf("%d\t",order[m]);
          if(++m%4 == 0)
              printf("\n");
      }
      printf("ProposalsForOneImage-[order] before\n");
        for(int m=0;m<limit;){
          printf("%d\t",order[m]);
          if(++m%4 == 0)
              printf("\n");
      }
  }

  if (pre_nms_topn_ <= 0 || pre_nms_topn_ >= scores.size()) {
    // 4. sort all (proposal, score) pairs by score from highest to lowest
    // 5. take top pre_nms_topN (e.g. 6000)
    printf("ProposalsForOneImage-[sort type 1]\n");
    std::sort(order.begin(), order.end(), [&scores](int lhs, int rhs) {
      return scores[lhs] > scores[rhs];
    });
  } else {
    // Avoid sorting possibly large arrays; First partition to get top K
    // unsorted and then sort just those (~20x faster for 200k scores)
    printf("ProposalsForOneImage-[sort type 2]\n");
    std::partial_sort(
        order.begin(),
        order.begin() + pre_nms_topn_,
        order.end(),
        [&scores](int lhs, int rhs) { return scores[lhs] > scores[rhs]; });
    order.resize(pre_nms_topn_);
  }
  if(DEBUG){
      printf("ProposalsForOneImage-[order] after\n");
        for(int m=0;m<limit;){
          printf("%d\t",order[m]);
          if(++m%4 == 0)
              printf("\n");
      }
  }
  caffe2::ERArrXXf bbox_deltas_sorted;
  caffe2::ERArrXXf all_anchors_sorted;
  caffe2::EArrXf scores_sorted;
  caffe2::utils::GetSubArrayRows(
      bbox_deltas, caffe2::utils::AsEArrXt(order), &bbox_deltas_sorted);
  caffe2::utils::GetSubArrayRows(
      all_anchors.array(), caffe2::utils::AsEArrXt(order), &all_anchors_sorted);
  caffe2::utils::GetSubArray(scores, caffe2::utils::AsEArrXt(order), &scores_sorted);

  // Transform anchors into proposals via bbox transformations
  static const std::vector<float> bbox_weights{1.0, 1.0, 1.0, 1.0};
  printf("ProposalsForOneImage-[nms_thresh, post_nms_topN] : [%.5f, %d] \n",nms_thresh, post_nms_topN);
  auto proposals = caffe2::utils::bbox_transform(
      all_anchors_sorted,
      bbox_deltas_sorted,
      bbox_weights,
      caffe2::utils::BBOX_XFORM_CLIP_DEFAULT,
      correct_transform_coords_,
      angle_bound_on_,
      angle_bound_lo_,
      angle_bound_hi_);


  if(!DEBUG){  
      printf("ProposalsForOneImage-[bbox_deltas_sorted]\n");
        for(int m=0;m<limit;){
          printf("%f\t",bbox_deltas_sorted.data()[m]);
          if(++m%4 == 0)
              printf("\n");
      }

    printf("ProposalsForOneImage-[proposals before NMS] shape : (%d, %d)\n",proposals.rows(),proposals.cols());
    printf("ProposalsForOneImage-[proposals]\n");
    for(int m=0;m<limit;){
      printf("%.5f\t",proposals.data()[m]);
      if(++m%4 == 0)
          printf("\n");
    }
  }
  // 2. clip proposals to image (may result in proposals with zero area
  // that will be removed in the next step)
  proposals =caffe2::utils::clip_boxes(proposals, im_info[0], im_info[1], clip_angle_thresh_);

  // 3. remove predicted boxes with either height or width < min_size
  auto keep = caffe2::utils::filter_boxes(proposals, min_size, im_info);
  //DCHECK_LE(keep.size(), scores_sorted.size());

  // 6. apply loose nms (e.g. threshold = 0.7)
  // 7. take after_nms_topN (e.g. 300)
  // 8. return the top proposals (-> RoIs top)
  if (post_nms_topN > 0 && post_nms_topN < keep.size()) {
    keep = caffe2::utils::nms_cpu(proposals, scores_sorted, keep, nms_thresh, post_nms_topN);
  } else {
    keep = caffe2::utils::nms_cpu(proposals, scores_sorted, keep, nms_thresh);
  }

  // Generate outputs
  caffe2::utils::GetSubArrayRows(proposals, caffe2::utils::AsEArrXt(keep), out_boxes);
  caffe2::utils::GetSubArray(scores_sorted, caffe2::utils::AsEArrXt(keep), out_probs); 

  printf("ProposalsForOneImage-[proposals after NMS] shape : (%d, %d)\n",out_boxes->rows(),out_boxes->cols());
  printf("ProposalsForOneImage-[proposals after NMS]\n");
  for(int m=0;m<limit;){
    printf("%.5f\t",out_boxes->data()[m]);
    if(++m%4 == 0)
        printf("\n");
  }

}///////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
void GenerateProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { 
 
  GenerateProposalsParameter proposal_param = this->layer_param_.proposal_param(); 

  spatial_scale_= proposal_param.spatial_scale();
  nms_thresh_   = proposal_param.nms_thresh();
  pre_nms_topn_ = proposal_param.pre_nms_topn();
  min_size_     = proposal_param.min_size();
  //rpn_post_nms_topn_     = proposal_param.rpn_post_nms_topn();
  post_nms_topn_           = proposal_param.post_nms_topn();
  correct_transform_coords_= proposal_param.correct_transform_coords(); 
}

template <typename Dtype>
void GenerateProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num_images = bottom[0]->shape(0);
  vector<int> top_shape_rois;
  vector<int> top_shape_rois_prob;
  top_shape_rois.push_back(num_images*1000);
  top_shape_rois.push_back(5);
  top_shape_rois_prob.push_back(num_images*1000);
  top_shape_rois_prob.push_back(1);
  top[0]->Reshape(top_shape_rois);
  top[1]->Reshape(top_shape_rois_prob);
}

template <typename Dtype>
void GenerateProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const float* scores = (const float*)bottom[0]->cpu_data();//Input(0);
  const float* bbox_deltas = (const float*)bottom[1]->cpu_data();//Input(1);
  const float* im_info_tensor = (const float*)bottom[2]->cpu_data();//Input(2);
  const float* anchors = (const float*)bottom[3]->cpu_data();//Input(3);
  float* out_rois_ptr = (float*)top[0]->mutable_cpu_data();//Output(0);
  float* out_rois_probs_ptr = (float*)top[1]->mutable_cpu_data();//Output(1);
  printf("==================================================GenerateProposalLayer Forward==================================================\n");
  CHECK_EQ(bottom[0]->shape().size(), 4)<<"score should be 4D blob";//, scores.ndim());
  const auto num_images = bottom[0]->shape(0);
  const auto A = bottom[0]->shape(1);
  const auto height = bottom[0]->shape(2);
  const auto width = bottom[0]->shape(3);
  const auto K = height * width;
  const auto box_dim = bottom[3]->shape(1);
  CHECK_EQ(box_dim,4 );//|| box_dim == 5);

  // bbox_deltas: (num_images, A * box_dim, H, W)
  CHECK_EQ(bottom[1]->shape(0),num_images)<<"bbox_deltas's dim(0) != num_images";
  CHECK_EQ(bottom[1]->shape(1),box_dim * A)<<"bbox_deltas's dim(1) != box_dim * A";
  CHECK_EQ(bottom[1]->shape(2),height)<<"bbox_deltas's dim(2) != height";
  CHECK_EQ(bottom[1]->shape(3),width)<<"bbox_deltas's dim(3) != width";

  // im_info_tensor: (num_images, 3), format [height, width, scale; ...]
  CHECK_EQ(bottom[2]->shape(0),num_images);
  CHECK_EQ(bottom[2]->shape(1),3);

  // anchors: (A, box_dim)
  CHECK_EQ(bottom[3]->shape(0),A);
  CHECK_EQ(bottom[3]->shape(1),box_dim);

  //Broadcast the anchors to all pixels
  //caffe2::Workspace workspace;
  //caffe2::CPUContext context; 
  printf("A %d K %d width %d height %d box_dim %d num_images %d\n", A , K, width, height, box_dim, num_images); 

  //printf("spatial_scale_ %.5f\nnms_thresh_ %.5f\npre_nms_topn_ %d\nmin_size_ %.5f\npost_nms_topn_ %d\ncorrect_transform_coords_ %d\n",spatial_scale_,nms_thresh_,pre_nms_topn_,min_size_,post_nms_topn_,correct_transform_coords_);
  //const caffe2::Tensor<caffe2::CPU>& anchors_tensor;
  //auto ptr=anchors_tensor.mutable_data<float>();
  //ptr[0]=anchors[0];  


  //(const caffe2::TensorCPU&)anchors
  //auto tcpu = wrk.
  auto all_anchors_vec =utils::ComputeAllAnchors(anchors, (int)A,(int)box_dim, height, width, 1.0f/spatial_scale_);// code in header

  if(DEBUG){
      printf("all_anchors_vec : \n");
      printf("shape : (%d, %d)\n",all_anchors_vec.rows(),all_anchors_vec.cols());
          
      for(int m=0;m<A*box_dim;){
        printf("%.5f ",all_anchors_vec.data()[m]);
        if(++m%box_dim == 0)
            printf("\n");
      }
  }

  Eigen::Map<const caffe2::ERMatXf> all_anchors(all_anchors_vec.data(), K * A, box_dim);

  Eigen::Map<const caffe2::ERArrXXf> im_info(
      im_info_tensor,//.data<float>(),
      bottom[2]->shape(0),
      bottom[2]->shape(1));

  const int roi_col_count = box_dim + 1; 

  std::vector<caffe2::ERArrXXf> im_boxes(num_images);
  std::vector<caffe2::EArrXf> im_probs(num_images); 

  for (int i = 0; i < num_images; i++) {
    auto cur_im_info = im_info.row(i);
    auto cur_bbox_deltas = utils::GetSubTensorView<float>(bbox_deltas, (const std::vector<int>)bottom[1]->shape(), bottom[1]->count(),i);// code in header
    auto cur_scores = utils::GetSubTensorView<float>(scores, (const std::vector<int>)bottom[0]->shape(), bottom[0]->count(), i);// code in header
    //12, 56, 56 -> 12, 28, 28 -> 12, 14, 14 -> 12, 7, 7
    //3 , 56, 56 -> 3 , 28, 28 -> 3 , 14, 14 -> 3 , 7, 7
    caffe2::ERArrXXf& im_i_boxes = im_boxes[i];
    caffe2::EArrXf& im_i_probs = im_probs[i]; 
    

    if(DEBUG){
        printf("[cur_bbox_deltas]\nsize : %d\n",cur_bbox_deltas.size());
        printf("shape : (");
        for(int n=0;n<cur_bbox_deltas.ndim();n++)
          printf("%d ",cur_bbox_deltas.dim(n));

        printf(")\n");

        int limit = 40;// cur_bbox_deltas.size();

        printf("[bbox_deltas]\n");
        for(int m=0;m<limit;){
          printf("%.5f\t",bbox_deltas[m]);
          if(++m%5 == 0)
              printf("\n");
        }
        //printf("[cur_bbox_deltas]\n");        
        //for(int m=0;m<limit;){
        //  printf("%.5f\t",cur_bbox_deltas.data()[m]);
        //  if(++m%5 == 0)
        //      printf("\n");
        //}

        printf("[all_anchors]\n");   
        printf("shape : (%d, %d)\n",all_anchors.rows(),all_anchors.cols());   
        for(int m=0;m<limit;){
          printf("%.5f\t",all_anchors.data()[m]);
          if(++m%5 == 0)
              printf("\n");
        }   
 
    }
    ProposalsForOneImage(// code above
                    cur_im_info,
                    all_anchors,
                    cur_bbox_deltas,
                    cur_scores,
                    &im_i_boxes,
                    &im_i_probs);
  }

  int roi_counts = 0;
  for (int i = 0; i < num_images; i++) {
    roi_counts += im_boxes[i].rows();
  }
  //out_rois->Extend(roi_counts, 50, &context_);
  //out_rois_probs->Extend(roi_counts, 50, &context_);
  //float* out_rois_ptr = out_rois;//->template mutable_data<float>();
  //float* out_rois_probs_ptr = out_rois_probs;//->template mutable_data<float>();
  int st_roi_idx=0;
  int st_roi_prob_idx=0;
  for (int i = 0; i < num_images; i++) {
    const caffe2::ERArrXXf& im_i_boxes = im_boxes[i];
    const caffe2::EArrXf& im_i_probs = im_probs[i];
    int csz = im_i_boxes.rows();
    //To DO : check check why its H/W is 56, 28, 14, 7!! caffe2 : 200, 100, 50, 25!!

    // write rois
    //Eigen::Map<caffe2::ERArrXXf> cur_rois(out_rois_ptr, csz, roi_col_count);
    //cur_rois.col(0).setConstant(i);
    //cur_rois.block(0, 1, csz, box_dim) = im_i_boxes;

    // write rois_probs
    //Eigen::Map<caffe2::EArrXf>(out_rois_probs_ptr, csz) = im_i_probs;
    int j=0;
    int k=0;
    for( ;  j < (im_i_boxes.rows())*(im_i_boxes.cols()+1)  ;  j++){
        if(j%5==0)
          out_rois_ptr[j+st_roi_idx]=(float)i;
        else
          out_rois_ptr[j+st_roi_idx]=im_boxes[i].data()[k++];
    }

    
/*
0 1 2 3     4 5 6 7     8  9 10 11
x y x y     x y x y     x  y x  y
0 1 2 3 4   5 6 7 8 9   10 11 12 13 14  15 16 17 18 19
0 x y x y   0 x y x y   0  x  y  x  y   0  x  y  x  y  
*/
    for(j=0;j<im_i_probs.rows()*im_i_probs.cols();j++){
          out_rois_probs_ptr[j+st_roi_prob_idx]=im_probs[i].data()[j];
    }
    st_roi_idx += csz * roi_col_count;
    st_roi_prob_idx += csz;
  }

  CHECK_EQ(im_boxes[0].rows(),im_probs[0].rows());

  vector<int> top_shape_rois;
  vector<int> top_shape_rois_prob;
  //printf("\nim_boxes[0].rows():%d\n",im_boxes[0].rows());//
  //printf("\nim_probs[0].rows():%d\n",im_probs[0].rows());//
  top_shape_rois.push_back(im_boxes[0].rows());
  top_shape_rois.push_back(5);
  top_shape_rois_prob.push_back(im_probs[0].rows());
  top_shape_rois_prob.push_back(1);
  top[0]->Reshape(top_shape_rois);
  top[1]->Reshape(top_shape_rois_prob);
  //printf("\nproposal counts :%d\n",top[0]->shape(0));//
  //printf("\nproposal prob counts :%d\n",top[1]->shape(0));//



  printf("generate proposals done\n");
}


#ifdef CPU_ONLY
STUB_GPU(GenerateProposalLayer);
#endif

INSTANTIATE_CLASS(GenerateProposalLayer);
REGISTER_LAYER_CLASS(GenerateProposal);
}

