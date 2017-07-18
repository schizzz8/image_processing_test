#include <opencv/cv.hpp>
#include <iostream>

#include "srrg_path_map/path_map.h"
#include "srrg_path_map/distance_map_path_search.h"
#include "srrg_system_utils/system_utils.h"

using namespace std;
using namespace srrg_core;

typedef std::vector< std::vector<Eigen::Vector2f> > ClusterVector;

int threshold = 50;
const int threshold_max = 200;

int type = cv::MORPH_RECT;
int size = 10;
cv::Mat element = cv::getStructuringElement(type,
                                            cv::Size (2*size + 1, 2*size + 1),
                                            cv::Point (size, size));

UnsignedCharImage marker;
UnsignedCharImage mask;
UnsignedCharImage bitmask;
cv::Mat output_image;
cv::Mat cluster_image;


string window_name = "reconstruction parameters";

int rows,cols;

void onTrackbar(int,void*){

    for (int r=0; r<rows; ++r){
        const unsigned char* mask_ptr = mask.ptr<const unsigned char>(r);
        const unsigned char* bitmask_ptr = bitmask.ptr<const unsigned char>(r);
        unsigned char* marker_ptr = marker.ptr<unsigned char>(r);
        for (int c=0; c<cols; ++c, ++mask_ptr, ++bitmask_ptr, ++marker_ptr) {
            const unsigned char& mask=*mask_ptr;
            const unsigned char& bitmask = *bitmask_ptr;
            if(bitmask == 255){
                *marker_ptr = mask - threshold;
            }
        }
    }

    cv::Mat dilated_image;
    bool iterate = true;
    while(iterate){
        cv::dilate(marker, dilated_image, element);
        cv::Mat min_image;
        cv::min(dilated_image,mask,min_image);
        cv::Mat diff_image;
        cv::subtract(min_image,marker,diff_image);
        int diff = cv::countNonZero(diff_image);
        marker = min_image;
        if (diff == 0)
            iterate = false;
    }

    cv::subtract(mask,marker,output_image);
    cv::imshow(window_name, output_image);

}

int main(int argc, char** argv)
{
    string input_filename = argv[1];

    cv::Mat input = cv::imread(input_filename, 1);
    rows=input.rows;
    cols=input.cols;

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

    mask.create(rows,cols);
    mask = 100;
    bitmask.create(rows,cols);
    bitmask = 0;
    for (int r=0; r<rows; ++r){
        const float* dist_ptr = distances.ptr<const float>(r);
        const unsigned char* occ_ptr = occupancy.ptr<const unsigned char>(r);
        unsigned char* mask_ptr = mask.ptr<unsigned char>(r);
        unsigned char* bitmask_ptr = bitmask.ptr<unsigned char>(r);
        for (int c=0; c<cols; ++c, ++mask_ptr, ++bitmask_ptr, ++dist_ptr, ++occ_ptr) {
            const float& dist=*dist_ptr;
            const unsigned char& occ = *occ_ptr;
            if(occ == 254){
                *mask_ptr=(std::sqrt(dist)/mdist)*255;
                *bitmask_ptr = 255;
            }
        }
    }
    marker.create(rows,cols);
    marker = 100;

    cv::namedWindow("distance map",CV_WINDOW_NORMAL);
    cv::imshow("distance map", mask);

    cv::waitKey(0);


    cv::namedWindow(window_name,CV_WINDOW_NORMAL);
    cv::createTrackbar("trackbar",window_name,&threshold,threshold_max,onTrackbar);
    onTrackbar(threshold,0);

    cv::waitKey(0);

    cv::imwrite("output_image.png",output_image);

    int count=0;
    float radius = 0.5f;
    float resolution = 0.01;
    ClusterVector clusters;
    vector<Eigen::Vector2f> points;

    for(int r=0; r<rows; ++r)
        for(int c=0; c<cols; ++c){

            if(output_image.at<unsigned char>(r,c) > 9)
                points.push_back(Eigen::Vector2f (c*resolution,r*resolution));
        }

    cerr << "Clustering of " << points.size() << " points" << endl;
    vector<bool> processed (points.size(),false);

    for(int i=0; i<points.size(); i++){
        if(processed[i])
            continue;

        vector<int> seed_queue;
        int sq_idx = 0;
        seed_queue.push_back(i);

        processed[i] = true;

        while(sq_idx < seed_queue.size()){
            vector<int> neighbors;

            for(int j=0; j<points.size(); j++){
                if(j == i)
                    continue;

                //cerr << (points[i] - points[j]).norm() << endl;

                if((points[i] - points[j]).norm() <= radius)
                    neighbors.push_back(j);
            }

            if(neighbors.size() == 0){
                sq_idx++;
                continue;
            }

            for(int j=0; j<neighbors.size(); j++){
                if(processed[neighbors[j]])
                    continue;
                seed_queue.push_back(neighbors[j]);
                processed[neighbors[j]] = true;
            }
            sq_idx++;
        }

        vector<Eigen::Vector2f> cluster;
        for(int j=0; j < seed_queue.size(); j++){
            cluster.push_back(points[seed_queue[j]]);
        }
        count++;
        clusters.push_back(cluster);
    }

    cerr << "Found " << count << " clusters" << endl;

    mask.copyTo(cluster_image);
    cv::cvtColor(cluster_image,cluster_image,CV_GRAY2RGB);

    vector<cv::Scalar> colors;
    colors.push_back(cv::Scalar (255,0,0));
    colors.push_back(cv::Scalar (0,255,0));
    colors.push_back(cv::Scalar (0,0,255));

    for(int ii=0; ii<clusters.size(); ii++){
        vector<Eigen::Vector2f> cluster = clusters[ii];
        for(vector<Eigen::Vector2f>::const_iterator it = cluster.begin();
            it != cluster.end();
            it++){
            Eigen::Vector2f point = *it;
            cv::circle(cluster_image,
                       cv::Point ((point.x())/resolution,(point.y())/resolution),
                       8,
                       colors[ii%3],
                    2
                    );
        }
    }

    cv::namedWindow("Image",cv::WINDOW_NORMAL);
    cv::imshow("Image",cluster_image);

    cv::waitKey(0);


    return 0;
}
