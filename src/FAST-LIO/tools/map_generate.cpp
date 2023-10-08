#include <pcl/ModelCoefficients.h>
#include <iostream>
#include <cmath>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>

#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/moment_of_inertia_estimation.h>

#include <pcl/common/concatenate.h>

#include <string>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <visualization_msgs/Marker.h>



typedef pcl::PointXYZ PointT;


class MapGenerate
{
  public:
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
      pcl::NormalEstimation<PointT, pcl::Normal> ne;
      pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
      pcl::ExtractIndices<PointT> extract;
      pcl::ExtractIndices<pcl::Normal> extract_normals;
      pcl::search::KdTree<PointT>::Ptr tree;

      pcl::PointCloud<PointT>::Ptr cloud_filtered;
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals;
      pcl::PointCloud<PointT>::Ptr cloud_filtered2;
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2;
      pcl::ModelCoefficients::Ptr coefficients_plane, coefficients_cylinder;
      pcl::PointIndices::Ptr inliers_plane, inliers_cylinder;
  
      pcl::PointCloud<PointT>::Ptr cloud_plane;
      pcl::PointCloud<PointT>::Ptr cloud_cylinder;
      ros::NodeHandle nh;
      // 直通
      double min_x, max_x, min_y, max_y, min_z, max_z;
      // 体素
      float voxel_size_x, voxel_size_y, voxel_size_z;
      ros::Subscriber pt_sub;
      ros::Publisher marker_pub;
      visualization_msgs::Marker bbox_marker;

  MapGenerate();
  void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
      pcl::fromROSMsg(*msg, *cloud);
      bbox_marker.header.frame_id = msg->header.frame_id; // 设置坐标系
      bbox_marker.header.stamp = msg->header.stamp;
      bbox_marker.action = visualization_msgs::Marker::ADD;
      bbox_marker.scale.x = 0.01; // 线的宽度
      bbox_marker.id = 0;
      bbox_marker.type = visualization_msgs::Marker::CUBE;
      // Set the color (RGBA) of the bounding box
      bbox_marker.color.r = 1.0;
      bbox_marker.color.g = 0.0;
      bbox_marker.color.b = 0.0;
      bbox_marker.color.a = 0.5; // Set the transparency
      cloudPre();
      normals_estimate();
      // 分割平面
      plane_seg();
      // 保存平面
      get_plane();
      // 移除平面
      remove_plane();
      // 圆柱分割
      cylinder_seg();
      // 获取圆柱
      get_cylinder();
      marker_pub.publish(bbox_marker);
  }
  void cloudPre();
  void normals_estimate()
  {
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud_filtered);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);
  }

  void plane_seg()
  {
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.05); // 0.03
    seg.setInputCloud(cloud_filtered);
    seg.setInputNormals(cloud_normals);
    seg.segment(*inliers_plane, *coefficients_plane);
    std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;
  }

  void get_plane()
  {
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers_plane);
    extract.setNegative(false);
    extract.filter(*cloud_plane);
    // pcl::PCDWriter writer;
    // std::string file_name = std::string("plane.pcd");
    // std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") + file_name);
    // writer.write<pcl::PointXYZ> (all_points_dir, *cloud, false);
  }

  void remove_plane()
  {
    extract.setNegative(true);
    extract.filter(*cloud_filtered2);
    extract_normals.setNegative(true);
    extract_normals.setInputCloud(cloud_normals);
    extract_normals.setIndices(inliers_plane);
    extract_normals.filter(*cloud_normals2);
  }

  void cylinder_seg()
  {
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(0.15); // 0.05
    seg.setRadiusLimits(0.0, 1.5);
    seg.setInputCloud(cloud_filtered2);
    seg.setInputNormals(cloud_normals2);

    // 圆柱体轴线在X轴方向上的位置
    // 圆柱体轴线在Y轴方向上的位置
    // 圆柱体轴线在Z轴方向上的位置
    // 表示圆柱体在X轴方向上的朝向（单位向量）
    // 表示圆柱体在Y轴方向上的朝向（单位向量）
    // 表示圆柱体在Z轴方向上的朝向（单位向量）
    // 圆柱体的半径
  
    seg.segment(*inliers_cylinder, *coefficients_cylinder);
    std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;
  }

  void get_cylinder()
  {
    extract.setInputCloud(cloud_filtered2);
    extract.setIndices(inliers_cylinder);
    extract.setNegative(false);
    
    extract.filter(*cloud_cylinder);

    if (cloud_cylinder->points.empty())
      std::cerr << "Can't find the cylindrical component." << std::endl;
    // else
    // {
    //   std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size() << " data points." << std::endl;
    //   pcl::PCDWriter writer;
    //   std::string file_name = std::string("cylinder.pcd");
    //   std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") + file_name);
    //   writer.write<pcl::PointXYZ> (all_points_dir, *cloud_cylinder, false);
    // }
  }
  void generateCylinderPointCloud(const double z_resolution,const double theta_num,const double diff_radius)
  {
    // 先构建轴线为Z轴的圆柱面点云
    int Num = theta_num;
    float inter = 2.0 * M_PI / Num;
    Eigen::RowVectorXd vectorx(Num), vectory(Num), vectorz(Num);
    Eigen::RowVector3d axis(coefficients_cylinder->values[3], coefficients_cylinder->values[4], coefficients_cylinder->values[5]);
    float length = axis.norm();
    vectorx.setLinSpaced(Num, 0, Num - 1);
    vectory = vectorx;
    float x0, y0, z0,r0;
    x0 = coefficients_cylinder->values[0];
    y0 = coefficients_cylinder->values[1];
    z0 = coefficients_cylinder->values[2];
    r0 = coefficients_cylinder->values[6] + diff_radius;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr raw(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr final(new pcl::PointCloud<pcl::PointXYZINormal>);

    for (float z(-10); z <= 10; z += z_resolution)
    {
      for (auto i = 0; i < Num; ++i) {
        pcl::PointXYZINormal point;
        point.x = r0 * cos(vectorx[i] * inter);
        point.y = r0 * sin(vectory[i] * inter);
        point.z = z;
        point.normal_x = r0;
        raw->points.push_back(point);
      }
      pcl::PointXYZINormal point;
      point.x = 0;
      point.y = 0;
      point.z = z;
      raw->points.push_back(point);
    }
    
    raw->width = (int)raw->size();
    raw->height = 1;
    raw->is_dense = false;


    // 点云旋转 Z轴转到axis
    Eigen::RowVector3d  Z(0.0, 0.0, 0.1), T0(0, 0, 0), T(coefficients_cylinder->values[0], coefficients_cylinder->values[1], coefficients_cylinder->values[2]);
    Eigen::Matrix3d R;
    Eigen::Matrix3d E = Eigen::MatrixXd::Identity(3, 3);
    Eigen::Matrix4d Rotate,Translation;
    R = Eigen::Quaterniond::FromTwoVectors(Z, axis).toRotationMatrix();
    Rotate.setIdentity();
    Translation.setIdentity();

    // 旋转
    Rotate.block<3, 3>(0, 0) = R;
    Rotate.block<3, 1>(0, 3) = T;
    pcl::transformPointCloud(*raw, *tmp, Rotate);

    // 筛选
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cloud_cylinder, min_pt, max_pt);

    pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud (cloud_cylinder);
    feature_extractor.compute ();
    pcl::PointXYZ min_point_OBB;
    pcl::PointXYZ max_point_OBB;
    pcl::PointXYZ position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    // Set the position and orientation of the bounding box
    bbox_marker.pose.position.x = position_OBB.x;
    bbox_marker.pose.position.y = position_OBB.y;
    bbox_marker.pose.position.z = position_OBB.z;
    // Set the orientation as a Quaternion
    Eigen::Quaternionf rotation_quaternion(rotational_matrix_OBB);
    bbox_marker.pose.orientation.x = rotation_quaternion.x();;
    bbox_marker.pose.orientation.y = rotation_quaternion.y();;
    bbox_marker.pose.orientation.z = rotation_quaternion.z();;
    bbox_marker.pose.orientation.w = rotation_quaternion.w();;

    // Set the scale of the bounding box
    bbox_marker.scale.x = max_point_OBB.x - min_point_OBB.x;
    bbox_marker.scale.y = max_point_OBB.y - min_point_OBB.y;
    bbox_marker.scale.z = max_point_OBB.z - min_point_OBB.z; 
  }
};


MapGenerate::MapGenerate()
{
    cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    tree.reset(new pcl::search::KdTree<PointT>());
    cloud_filtered.reset(new pcl::PointCloud<PointT>);
    cloud_normals.reset(new pcl::PointCloud<pcl::Normal>);
    cloud_filtered2.reset(new pcl::PointCloud<PointT>);
    cloud_normals2.reset(new pcl::PointCloud<pcl::Normal>);
    coefficients_plane.reset(new pcl::ModelCoefficients);
    coefficients_cylinder.reset(new pcl::ModelCoefficients);
    inliers_plane.reset(new pcl::PointIndices);
    inliers_cylinder.reset(new pcl::PointIndices);
    cloud_plane.reset(new pcl::PointCloud<PointT>());
    cloud_cylinder.reset(new pcl::PointCloud<PointT>());
    nh.param<double>("max_x",max_x,5.0);
    nh.param<double>("min_x",min_x,0.0); 
    nh.param<double>("max_y",max_y,2.0);
    nh.param<double>("min_y",min_y,-2.0); 
    nh.param<double>("max_z",max_z,3.0);
    nh.param<double>("min_z",min_z,0.0);
    pt_sub = nh.subscribe("/cloud_registered", 1, &MapGenerate::pointCloudCallback, this);
    marker_pub = nh.advertise<visualization_msgs::Marker>("aim", 1);
}

void MapGenerate::cloudPre()
{
    // 点云直通滤波
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (min_x, max_x);
    pass.filter (*cloud);

    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (min_y, max_y);
    pass.filter (*cloud);

    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (min_z, max_z);
    pass.filter (*cloud);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_filtered);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "map_generate");
    MapGenerate mg;
    ros::spin();
}


