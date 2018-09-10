#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {


template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};


template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T> >& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < (T)-1.0 || y > (T)height || x < (T)-1.0 || x > (T)width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
          } else {
            x_high = x_low + 1;
          }

          T ly = y - y_low;
          T lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
          
          // save weights and indeces
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void ROIAlignForward(const int nthreads,
                    const T* bottom_data,
                    const T& spatial_scale,
                    const int channels,const int height,const int width,
                    const int pooled_height,const int pooled_width,
                    const int sampling_ratio,
                    const T* bottom_rois,
                    int roi_cols,
                    T* top_data) {
  CHECK((roi_cols ==4)|| (roi_cols == 5)) << "roi_cols should be 4 or 5";

  int n_rois = nthreads / channels / pooled_width / pooled_height;
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    // roi could have 4 or 5 columns
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    int roi_batch_ind = 0;
    if (roi_cols == 5) {
      roi_batch_ind = offset_bottom_rois[0];
      offset_bottom_rois++; 
    }
    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_bottom_rois[0] * spatial_scale;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale;
    // T roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[3] * spatial_scale);

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width); 
    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width); 
    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    // we want to precalculate indeces and weights shared by all chanels,
    // this is the key point of optimiation 
    std::vector<PreCalc<T> > pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
 
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);

      for (int c = 0; c < channels; c++) {
        int index_n_c = index_n + c * pooled_width * pooled_height;
        const T* offset_bottom_data =
            bottom_data + (roi_batch_ind * channels + c) * height * width;
        int pre_calc_index = 0;
          
          for (int ph = 0; ph < pooled_height; ph++) {
            for (int pw = 0; pw < pooled_width; pw++) {
              int index = index_n_c + ph * pooled_width + pw;

                  T output_val = 0.;
                  for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                    for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                      PreCalc<T> pc = pre_calc[pre_calc_index];
                      //printf("PreCalc\n");
                      output_val += pc.w1 * offset_bottom_data[pc.pos1] +
                          pc.w2 * offset_bottom_data[pc.pos2] +
                          pc.w3 * offset_bottom_data[pc.pos3] +
                          pc.w4 * offset_bottom_data[pc.pos4];
                          //printf("output_val\n");

                      pre_calc_index += 1;
                    }
                  }
                  output_val /= count;

                  top_data[index] = output_val;
            } // for pw
          } // for ph
      } // for c 
  } // for n
}

template <typename Dtype>
void RoIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { 

    ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param(); 

    sampling_ratio = roi_align_param.sampling_ratio();
    pooled_w       = roi_align_param.pooled_w();
    pooled_h       = roi_align_param.pooled_h();
    spatial_scale  = roi_align_param.spatial_scale();
}

template <typename Dtype>
void RoIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    
    top[0]->Reshape(bottom[1]->shape(0),bottom[0]->shape(1),pooled_w,pooled_h);//bottom[0]->shape(1)
    // 841, 256, 7,7
    // 131, 256, 7,7
    // 27, 256, 7,7
    // 1, 256, 7,7                     
    // Concat -> 1000, 256, 7,7?
}

template <typename Dtype>
void RoIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  printf("==============================================RoIAlignLayer Forward start=====================================\n");

  const float* X =(const float*)bottom[0]->cpu_data();// Input data to pool, NCHW
  const float* R =(const float*)bottom[1]->cpu_data();// RoIs
  float* Y =(float*)top[0]->mutable_cpu_data();       // RoI pooled data
  if (bottom[1]->shape(0) == 0) {//number of elements
    // Handle empty rois
      top[0]->Reshape(0, bottom[0]->shape(1), pooled_h, pooled_w);// [0,x,y,z] possible ??
    // The following mutable_data calls are needed to allocate the tensors
    //Y->template mutable_data<float>();
    return ;
  }

  printf("X shape : %d, %d, %d, %d R shape : %d, %d \n", bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3), bottom[1]->shape(0), bottom[1]->shape(1));
  int i=0;
  printf("%.2f %.2f %.2f %.2f %.2f\n",R[0],R[1],R[2],R[3],R[4]);
  //X shape : 1, 256, 56, 56 R shape : 841, 5 
  //X shape : 1, 256, 56, 56 R shape : 131, 5 
  //X shape : 1, 256, 56, 56 R shape : 27, 5 
  //X shape : 1, 256, 56, 56 R shape : 1, 5 


  CHECK_EQ(bottom[1]->shape().size(), 2);//
  // if R has 5 columns, the first column is the index, otherwise 0
  CHECK_EQ(bottom[1]->shape(1) , 5);//

  CHECK_GT(sampling_ratio , 0);
  
  top[0]->Reshape(bottom[1]->shape(0), bottom[0]->shape(1), pooled_h, pooled_w);
  int output_size = top[0]->count();
  
  ROIAlignForward<float>(output_size,
                          X,
                          spatial_scale,
                          bottom[0]->shape(1),
                          bottom[0]->shape(2),
                          bottom[0]->shape(3),
                          pooled_h,
                          pooled_w,
                          sampling_ratio,
                          R,
                          bottom[1]->shape(1),
                          Y
                          );
  printf("Y shape : %d, %d, %d, %d \n", top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));
  printf("==============================================RoIAlignLayer Forward end=======================================\n");
}


#ifdef CPU_ONLY
STUB_GPU(RoIAlignLayer);
#endif

INSTANTIATE_CLASS(RoIAlignLayer);
REGISTER_LAYER_CLASS(RoIAlign);
}

