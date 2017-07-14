#include <opencv/cv.hpp>
#include <iostream>

#include "srrg_path_map/path_map.h"
#include "srrg_path_map/distance_map_path_search.h"
#include "srrg_system_utils/system_utils.h"

using namespace std;
using namespace srrg_core;

int main(int argc, char** argv)
{
    string input_filename = argv[1];

    cv::Mat input = cv::imread(input_filename, 1);
    int rows=input.rows;
    int cols=input.cols;

    UnsignedCharImage occupancy;
    cv::cvtColor(input,occupancy,CV_BGR2GRAY);

    IntImage points_image;
    points_image.create(rows,cols);
    points_image=-1;

    int id=0;
    for(int r=0; r<rows; ++r){
        const unsigned char* occ_ptr = occupancy.ptr<const unsigned char>(r);
        int* idx_ptr = points_image.ptr<int>(r);
        for(int c=0; c<cols; ++c, ++occ_ptr, ++idx_ptr, ++id){
            const unsigned char& occ = *occ_ptr;
            int& idx = *idx_ptr;
            if(occ == 0)
                idx = id;
        }
    }

    PathMap distance_map;
    DistanceMapPathSearch dmap_calculator;
    int d_max = 100;
    dmap_calculator.setOutputPathMap(distance_map);
    dmap_calculator.setIndicesImage(points_image);
    dmap_calculator.setMaxDistance(d_max);
    dmap_calculator.init();
    dmap_calculator.compute();

    FloatImage distances = dmap_calculator.distanceImage();
    float mdist = 0;
    for (int r=0; r<rows; r++){
        const float* dist_ptr = distances.ptr<const float>(r);
        for (int c=0; c<cols; c++, dist_ptr++){
            const float& dist = *dist_ptr;
            if (dist == std::numeric_limits<float>::max())
                continue;
            mdist = (mdist < dist) ?  dist : mdist;
        }
    }
    mdist = std::sqrt(mdist);

    cerr << "max computed distance: " << mdist << endl;

    RGBImage output_image;
    output_image.create(rows,cols);
    output_image = cv::Vec3b(100,100,100);
    for (int r=0; r<rows; ++r){
        const float* dist_ptr = distances.ptr<const float>(r);
        const unsigned char* occ_ptr = occupancy.ptr<const unsigned char>(r);
        cv::Vec3b* out_ptr = output_image.ptr<cv::Vec3b>(r);
        for (int c=0; c<cols; ++c, ++out_ptr, ++dist_ptr, ++occ_ptr) {
            const float& dist=*dist_ptr;
            const unsigned char& occ = *occ_ptr;
            if(occ != 205){
                float ndist = std::sqrt(dist)/mdist;
                *out_ptr=cv::Vec3b(0,ndist*255,0);
            }
        }
    }

    cv::namedWindow("distance map",CV_WINDOW_NORMAL);
    cv::imshow("distance map", output_image);
    cv::waitKey();


    return 0;
}
