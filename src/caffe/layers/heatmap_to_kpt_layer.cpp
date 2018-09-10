#include <algorithm>
#include <cfloat>
#include <vector>
#include <stdio.h>

#include "caffe/layers/heatmap_to_kpt_layer.hpp"

//#include "opencv2/imgproc/imgproc.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

using namespace cv;
namespace caffe { 
void scores_to_probs(vector<float>scores){
    //Transforms CxHxW of scores to probabilities spatially.
    /*
    channels = scores.shape[0]
    for c in range(channels):
        temp = scores[c, :, :]
        max_score = temp.max()
        temp = np.exp(temp - max_score) / np.sum(np.exp(temp - max_score))
        scores[c, :, :] = temp
    return scores
    */
}
template <typename Dtype>
void HeatmapToKeypointsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {  
}

template <typename Dtype>
void HeatmapToKeypointsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape_keypoint; 
  int num_kpts=17;
    
  top_shape_keypoint.push_back(2*num_kpts);
  top_shape_keypoint.push_back(1);
  top[0]->Reshape(top_shape_keypoint);
}
template <typename Dtype>
void HeatmapToKeypointsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 
  printf("==============================================HeatmapToKeypointsLayer start=========================\n");
  const float *rois =(const float *)bottom[0]->cpu_data();   //'bbox_nms'  [K,5]       == box_idx, x1, y1, x2, y2
  const float *maps =(const float *)bottom[1]->cpu_data();   //'kps_score' [K,17,56,56]== box_idx

  CHECK_EQ(bottom[1]->shape(0),bottom[0]->shape(0));

  float *kps_vector =(float *)top[0]->mutable_cpu_data();//[17,2]
  printf("input size :[%d %d], [%d %d %d %d]\n",bottom[0]->shape(0), bottom[0]->shape(1), 
                                                bottom[1]->shape(0), bottom[1]->shape(1),bottom[1]->shape(2), bottom[1]->shape(3));  

  int num_classes=2;
  float offset_x;
  float offset_y; 
  //  offset_x = rois[:, 0]
  //  offset_y = rois[:, 1]
  for(int box_idx=0;box_idx<bottom[1]->shape(0);box_idx++){
      offset_x= rois[5*box_idx+1];
      offset_y= rois[5*box_idx+2];
      printf("offset_x, offset_y :[%.2f %.2f]\n",offset_x, offset_y);
      /*
      widths = rois[:, 2] - rois[:, 0]
      heights = rois[:, 3] - rois[:, 1]
      widths = np.maximum(widths, 1)
      heights = np.maximum(heights, 1)
      widths_ceil = np.ceil(widths)
      heights_ceil = np.ceil(heights)
      */
      float box_width  = rois[5*box_idx+3] - rois[5*box_idx+1];
      float box_height = rois[5*box_idx+4] - rois[5*box_idx+2];
      box_width = box_width >1 ? box_width : 1;
      box_height= box_height>1 ? box_height: 1;
      printf("box_width, box_height :[%.2f %.2f]\n",box_width, box_height);
      float box_width_ceil =std::ceil(box_width);
      float box_height_ceil=std::ceil(box_height);
      printf("box_width_ceil, box_height_ceil :[%.1f %.1f]\n",box_width_ceil, box_height_ceil);
      /*
      width_correction = widths[i] / widths_ceil[i]
      height_correction = heights[i] / heights_ceil[i]
      */
      float box_width_correction = box_width  / box_width_ceil;
      float box_height_correction= box_height / box_height_ceil;
      printf("box_width_correction, box_height_correction :[%.5f %.5f]\n",box_width_correction, box_height_correction);
      /*
      w = roi_map.shape[2]
      for k in range(cfg.KRCNN.NUM_KEYPOINTS):
          pos = roi_map[k, :, :].argmax()
          x_int = pos % w
          y_int = (pos - x_int) // w 
          x = (x_int + 0.5) * width_correction
          y = (y_int + 0.5) * height_correction
          xy_preds[i, 0, k] = x + offset_x[i]
          xy_preds[i, 1, k] = y + offset_y[i]  
      */ 
      vector<vector <int> > pos_xy=max_value_coordinate_xy(&maps[box_idx*17*56*56],17,56,56,box_width,box_height);
      for(int kpt_idx=0;kpt_idx<17;kpt_idx++){ // kpts for current box==box_idx          
          float x=(pos_xy[0][kpt_idx] + 0.5) * box_width_correction;
          float y=(pos_xy[1][kpt_idx] + 0.5) * box_height_correction;
          printf("x, y :[%.2f %.2f]\n",x, y);
          //vector< float > xy_coord{x + offset_x,y + offset_y};
          //xy_preds.push_back(xy_coord);
          //xy_preds = [K, 2, 17], K=1 -> [1,2,17] == [2,17]
          kps_vector[kpt_idx*2]  =x + offset_x;
          kps_vector[kpt_idx*2+1]=y + offset_y;
      }
  }                  
  printf("==============================================HeatmapToKeypointsLayer done==========================\n");
}
vector< vector<int> > max_value_coordinate_xy(const float* maps, int N, 
                                        int HEATMAP_W, int HEATMAP_H, 
                                        float origin_box_W, float origin_box_H){
  //33,9, 
  float max_value=0.0f;
  vector<vector <int> > ret;  
  vector<int> kpts_X;
  vector<int> kpts_Y;
  float cur_max_X=0.0f;
  float cur_max_Y=0.0f;
  float scale_x=origin_box_W/HEATMAP_W;
  float scale_y=origin_box_H/HEATMAP_H;
  printf("scale_x, scale_y :[%.2f %.2f]\n",scale_x, scale_y);
  for(int n=0;n<N;n++){
      for(int h=0;h<HEATMAP_H;h++){
          for(int w=0;w<HEATMAP_W;w++){
              if(maps[n*HEATMAP_H*HEATMAP_W+h*HEATMAP_W+w]>max_value){
                max_value=maps[n*HEATMAP_H*HEATMAP_W+h*HEATMAP_W+w];
                cur_max_X=w*scale_x;
                cur_max_Y=h*scale_y;
              }
          }
      max_value=0.0f;
      }
    printf("cur_max_X, cur_max_Y :[%.2f %.2f]\n",cur_max_X, cur_max_Y);
    kpts_X.push_back(cur_max_X);
    kpts_Y.push_back(cur_max_Y);
  }
  ret.push_back(kpts_X);
  ret.push_back(kpts_Y);

  return ret;
}
#ifdef CPU_ONLY
STUB_GPU(HeatmapToKeypointsLayer);
#endif

INSTANTIATE_CLASS(HeatmapToKeypointsLayer);
REGISTER_LAYER_CLASS(HeatmapToKeypoints);
}

