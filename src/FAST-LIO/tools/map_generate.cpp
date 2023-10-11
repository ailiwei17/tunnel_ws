#include <cmath>
#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/common/concatenate.h>

#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <visualization_msgs/Marker.h>

typedef pcl::PointXYZ PointT;

class MapGenerate {
public:
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
  pcl::ExtractIndices<PointT> extract;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::search::KdTree<PointT>::Ptr tree;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  pcl::PointCloud<PointT>::Ptr accumulated_cloud;

  pcl::PointCloud<PointT>::Ptr stable_cloud_filtered, moving_cloud_filtered;
  pcl::PointCloud<pcl::Normal>::Ptr stable_cloud_normals, moving_cloud_normals;
  pcl::PointCloud<PointT>::Ptr stable_cloud_filtered2, moving_cloud_filtered2;
  pcl::PointCloud<pcl::Normal>::Ptr stable_cloud_normals2, moving_cloud_normals2;
  pcl::ModelCoefficients::Ptr stable_coefficients_plane,
      stable_coefficients_cylinder, moving_coefficients_plane, moving_coefficients_cylinder;
  pcl::PointIndices::Ptr stable_inliers_plane, stable_inliers_cylinder, moving_inliers_plane, moving_inliers_cylinder;

  pcl::PointCloud<PointT>::Ptr stable_cloud_plane, stable_cloud_cylinder, moving_cloud_plane, moving_cloud_cylinder;

  visualization_msgs::Marker stable_bbox_marker, moving_bbox_marker;

  ros::NodeHandle nh;
  // 直通
  double stable_min_x, stable_max_x, stable_min_y, stable_max_y, stable_min_z,
      stable_max_z;
  double moving_min_x, moving_max_x, moving_min_y, moving_max_y, moving_min_z,
      moving_max_z;

  ros::Subscriber pt_sub;
  ros::Publisher marker_pub;
  
  int point_cloud_count = 0;

  // 拟合参数
  double min_radius;
  double max_radius;

  // 半径膨胀
  double diff_radius;

  MapGenerate();
  void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    pcl::fromROSMsg(*msg, *cloud);
    *accumulated_cloud += *cloud;
    point_cloud_count++;

    // 当接收到十次点云后执行操作
    if (point_cloud_count >= 20) {
      stable_bbox_marker.header.frame_id = msg->header.frame_id; // 设置坐标系
      stable_bbox_marker.header.stamp = msg->header.stamp;
      stable_bbox_marker.action = visualization_msgs::Marker::ADD;
      stable_bbox_marker.scale.x = 0.01; // 线的宽度
      stable_bbox_marker.id = 0;
      stable_bbox_marker.type = visualization_msgs::Marker::CYLINDER;
      // Set the color (RGBA) of the bounding box
      stable_bbox_marker.color.r = 1.0;
      stable_bbox_marker.color.g = 0.0;
      stable_bbox_marker.color.b = 0.0;
      stable_bbox_marker.color.a = 1.0; // Set the transparency

      moving_bbox_marker.header.frame_id = msg->header.frame_id; // 设置坐标系
      moving_bbox_marker.header.stamp = msg->header.stamp;
      moving_bbox_marker.action = visualization_msgs::Marker::ADD;
      moving_bbox_marker.scale.x = 0.01; // 线的宽度
      moving_bbox_marker.id = 1;
      moving_bbox_marker.type = visualization_msgs::Marker::CYLINDER;
      // Set the color (RGBA) of the bounding box
      moving_bbox_marker.color.r = 0.0;
      moving_bbox_marker.color.g = 0.0;
      moving_bbox_marker.color.b = 1.0;
      moving_bbox_marker.color.a = 1.0; // Set the transparency

      cloudPre(accumulated_cloud, stable_cloud_filtered, stable_min_x,
               stable_max_x, stable_min_y, stable_max_y, stable_min_z,
               stable_max_z);
      normals_estimate(tree, stable_cloud_filtered, stable_cloud_normals);
      plane_seg(stable_cloud_filtered, stable_cloud_normals,
                stable_inliers_plane, stable_coefficients_plane);
      get_plane(stable_cloud_filtered, stable_inliers_plane,
                stable_cloud_plane);
      remove_plane(stable_cloud_normals, stable_inliers_plane,
                   stable_cloud_filtered2, stable_cloud_normals2);
      cylinder_seg(stable_cloud_filtered2, stable_cloud_normals2,
                   stable_inliers_cylinder, stable_coefficients_cylinder);
      get_cylinder(stable_cloud_filtered, stable_cloud_filtered2,
                   stable_inliers_cylinder, stable_cloud_cylinder);
      

      cloudPre(accumulated_cloud, moving_cloud_filtered, moving_min_x,
               moving_max_x, moving_min_y, moving_max_y, moving_min_z,
               moving_max_z);
      normals_estimate(tree, moving_cloud_filtered, moving_cloud_normals);
      plane_seg(moving_cloud_filtered, moving_cloud_normals,
                moving_inliers_plane, moving_coefficients_plane);
      get_plane(moving_cloud_filtered, moving_inliers_plane,
                moving_cloud_plane);
      remove_plane(moving_cloud_normals, moving_inliers_plane,
                   moving_cloud_filtered2, moving_cloud_normals2);
      cylinder_seg(moving_cloud_filtered2, moving_cloud_normals2,
                   moving_inliers_cylinder, moving_coefficients_cylinder);
      get_cylinder(moving_cloud_filtered, moving_cloud_filtered2,
                   moving_inliers_cylinder, moving_cloud_cylinder);

      generateCylinderPointCloud(10, 0.3, 8, diff_radius,
                                 stable_coefficients_cylinder,
                                 stable_cloud_cylinder, stable_bbox_marker);

      generateCylinderPointCloud(10, 0.3, 8, diff_radius,
                                 moving_coefficients_cylinder,
                                 moving_cloud_cylinder, moving_bbox_marker);

      // 重置计数器和累积点云
      point_cloud_count = 0;
      accumulated_cloud->clear();
    }
  }
  void cloudPre(pcl::PointCloud<PointT>::Ptr input_cloud,
                pcl::PointCloud<PointT>::Ptr output_cloud, double min_x,
                double max_x, double min_y, double max_y, double min_z,
                double max_z);
  void normals_estimate(pcl::search::KdTree<PointT>::Ptr kd_tree,
                        pcl::PointCloud<PointT>::Ptr input_cloud,
                        pcl::PointCloud<pcl::Normal>::Ptr output_normals) {
    ne.setSearchMethod(kd_tree);
    ne.setInputCloud(input_cloud);
    ne.setKSearch(50);
    ne.compute(*output_normals);
  }

  void plane_seg(pcl::PointCloud<PointT>::Ptr input_cloud,
                 pcl::PointCloud<pcl::Normal>::Ptr input_normals,
                 pcl::PointIndices::Ptr output_inliers_plane,
                 pcl::ModelCoefficients::Ptr output_coefficients_plane) {
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.02); // 0.03
    seg.setInputCloud(input_cloud);
    seg.setInputNormals(input_normals);
    seg.segment(*output_inliers_plane, *output_coefficients_plane);
    std::cerr << "Plane coefficients: " << *output_coefficients_plane
              << std::endl;
  }

  void get_plane(pcl::PointCloud<PointT>::Ptr input_cloud,
                 pcl::PointIndices::Ptr input_inliers_plane,
                 pcl::PointCloud<PointT>::Ptr output_cloud_plane) {
    extract.setInputCloud(input_cloud);
    extract.setIndices(input_inliers_plane);
    extract.setNegative(false);
    extract.filter(*output_cloud_plane);
  }

  void remove_plane(pcl::PointCloud<pcl::Normal>::Ptr input_normals,
                    pcl::PointIndices::Ptr input_inliers_plane,
                    pcl::PointCloud<PointT>::Ptr output_cloud,
                    pcl::PointCloud<pcl::Normal>::Ptr output_normals) {
    extract.setNegative(true);
    extract.filter(*output_cloud);
    extract_normals.setNegative(true);
    extract_normals.setInputCloud(input_normals);
    extract_normals.setIndices(input_inliers_plane);
    extract_normals.filter(*output_normals);
  }

  void cylinder_seg(pcl::PointCloud<PointT>::Ptr input_cloud,
                    pcl::PointCloud<pcl::Normal>::Ptr input_normals,
                    pcl::PointIndices::Ptr output_inliers_cylinder,
                    pcl::ModelCoefficients::Ptr output_coefficients_cylinder) {
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(0.3); // 0.05
    seg.setRadiusLimits(min_radius, max_radius);
    // seg.setInputCloud(cloud_filtered);
    // seg.setInputNormals(cloud_normals);
    seg.setInputCloud(input_cloud);
    seg.setInputNormals(input_normals);

    // 圆柱体轴线在X轴方向上的位置
    // 圆柱体轴线在Y轴方向上的位置
    // 圆柱体轴线在Z轴方向上的位置
    // 表示圆柱体在X轴方向上的朝向（单位向量）
    // 表示圆柱体在Y轴方向上的朝向（单位向量）
    // 表示圆柱体在Z轴方向上的朝向（单位向量）
    // 圆柱体的半径

    seg.segment(*output_inliers_cylinder, *output_coefficients_cylinder);
    std::cerr << "Cylinder coefficients: " << *output_coefficients_cylinder
              << std::endl;
  }

  void get_cylinder(pcl::PointCloud<PointT>::Ptr input_cloud1,
                    pcl::PointCloud<PointT>::Ptr input_cloud2,
                    pcl::PointIndices::Ptr input_inliers_cylinder,
                    pcl::PointCloud<PointT>::Ptr output_cloud_cylinder) {
    extract.setInputCloud(input_cloud2);
    seg.setInputCloud(input_cloud1);
    extract.setIndices(input_inliers_cylinder);
    extract.setNegative(false);

    extract.filter(*output_cloud_cylinder);
  }
  void generateCylinderPointCloud(
      const double long_length, const double z_resolution,
      const double theta_num, const double diff_radius,
      pcl::ModelCoefficients::Ptr input_coefficients_cylinder,
      pcl::PointCloud<PointT>::Ptr input_cloud_cylinder,
      visualization_msgs::Marker input_bbox_marker) {
    // 先构建轴线为Z轴的圆柱面点云
    int Num = theta_num;
    float inter = 2.0 * M_PI / Num;
    Eigen::RowVectorXd vectorx(Num), vectory(Num), vectorz(Num);
    Eigen::RowVector3d axis(input_coefficients_cylinder->values[3],
                            input_coefficients_cylinder->values[4],
                            input_coefficients_cylinder->values[5]);
    float length = axis.norm();
    vectorx.setLinSpaced(Num, 0, Num - 1);
    vectory = vectorx;
    float x0, y0, z0, r0;
    x0 = input_coefficients_cylinder->values[0];
    y0 = input_coefficients_cylinder->values[1];
    z0 = input_coefficients_cylinder->values[2];
    r0 = input_coefficients_cylinder->values[6] + diff_radius;

    pcl::PointCloud<pcl::PointXYZ>::Ptr raw(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr final(
        new pcl::PointCloud<pcl::PointXYZ>);

    for (float z(-long_length); z <= long_length; z += z_resolution) {
      for (auto i = 0; i < Num; ++i) {
        pcl::PointXYZ point;
        point.x = r0 * cos(vectorx[i] * inter);
        point.y = r0 * sin(vectory[i] * inter);
        point.z = z;
        raw->points.push_back(point);
      }
      pcl::PointXYZ point;
      point.x = 0;
      point.y = 0;
      point.z = z;
      raw->points.push_back(point);
    }

    raw->width = (int)raw->size();
    raw->height = 1;
    raw->is_dense = false;

    // 点云旋转 Z轴转到axis
    Eigen::RowVector3d Z(0.0, 0.0, 0.1), T0(0, 0, 0),
        T(input_coefficients_cylinder->values[0],
          input_coefficients_cylinder->values[1],
          input_coefficients_cylinder->values[2]);
    Eigen::Matrix3d R;
    Eigen::Matrix3d E = Eigen::MatrixXd::Identity(3, 3);
    Eigen::Matrix4d Rotate, Translation;
    R = Eigen::Quaterniond::FromTwoVectors(Z, axis).toRotationMatrix();
    Rotate.setIdentity();
    Translation.setIdentity();

    // 旋转
    Rotate.block<3, 3>(0, 0) = R;
    Rotate.block<3, 1>(0, 3) = T;
    pcl::transformPointCloud(*raw, *tmp, Rotate);

    // 筛选
    pcl::PointXYZ min_pt, max_pt;
    if (!input_cloud_cylinder->points.empty()) {

      pcl::getMinMax3D(*input_cloud_cylinder, min_pt, max_pt);

      for (size_t i = 0; i < tmp->points.size(); ++i) {
        pcl::PointXYZ point = tmp->points[i];
        if (point.y < max_pt.y && point.y > min_pt.y) {
          final->push_back(point);
        }
      }

      if (!final->points.empty()) {
        pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
        feature_extractor.setInputCloud(final);
        feature_extractor.compute();
        pcl::PointXYZ min_point_OBB;
        pcl::PointXYZ max_point_OBB;
        pcl::PointXYZ position_OBB;
        Eigen::Matrix3f rotational_matrix_OBB;
        feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB,
                                 rotational_matrix_OBB);
        // Set the position and orientation of the bounding box
        input_bbox_marker.pose.position.x = position_OBB.x;
        input_bbox_marker.pose.position.y = position_OBB.y;
        input_bbox_marker.pose.position.z = position_OBB.z;
        // Set the orientation as a Quaternion
        Eigen::Matrix3f adjusted_rotational_matrix;
        adjusted_rotational_matrix.col(0) = rotational_matrix_OBB.col(1);
        adjusted_rotational_matrix.col(1) = rotational_matrix_OBB.col(2);
        adjusted_rotational_matrix.col(2) = rotational_matrix_OBB.col(0);
        Eigen::Quaternionf rotation_quaternion(adjusted_rotational_matrix);
        input_bbox_marker.pose.orientation.x = rotation_quaternion.x();
        input_bbox_marker.pose.orientation.y = rotation_quaternion.y();
        input_bbox_marker.pose.orientation.z = rotation_quaternion.z();
        input_bbox_marker.pose.orientation.w = rotation_quaternion.w();

        // Set the scale of the bounding box
        input_bbox_marker.scale.x = max_point_OBB.z - min_point_OBB.z;
        input_bbox_marker.scale.y = max_point_OBB.y - min_point_OBB.y;
        input_bbox_marker.scale.z = max_point_OBB.x - min_point_OBB.x;
        marker_pub.publish(input_bbox_marker);
      }
    }
  }
};

MapGenerate::MapGenerate() {
  cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
  accumulated_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
  tree.reset(new pcl::search::KdTree<PointT>());
  stable_cloud_filtered.reset(new pcl::PointCloud<PointT>);
  stable_cloud_normals.reset(new pcl::PointCloud<pcl::Normal>);
  stable_cloud_filtered2.reset(new pcl::PointCloud<PointT>);
  stable_cloud_normals2.reset(new pcl::PointCloud<pcl::Normal>);
  stable_coefficients_plane.reset(new pcl::ModelCoefficients);
  stable_coefficients_cylinder.reset(new pcl::ModelCoefficients);
  stable_inliers_plane.reset(new pcl::PointIndices);
  stable_inliers_cylinder.reset(new pcl::PointIndices);
  stable_cloud_plane.reset(new pcl::PointCloud<PointT>());
  stable_cloud_cylinder.reset(new pcl::PointCloud<PointT>());
  
  moving_cloud_filtered.reset(new pcl::PointCloud<PointT>);
  moving_cloud_normals.reset(new pcl::PointCloud<pcl::Normal>);
  moving_cloud_filtered2.reset(new pcl::PointCloud<PointT>);
  moving_cloud_normals2.reset(new pcl::PointCloud<pcl::Normal>);
  moving_coefficients_plane.reset(new pcl::ModelCoefficients);
  moving_coefficients_cylinder.reset(new pcl::ModelCoefficients);
  moving_inliers_plane.reset(new pcl::PointIndices);
  moving_inliers_cylinder.reset(new pcl::PointIndices);
  moving_cloud_plane.reset(new pcl::PointCloud<PointT>());
  moving_cloud_cylinder.reset(new pcl::PointCloud<PointT>());



  nh.param<double>("stable_max_x", stable_max_x, 5.0);
  nh.param<double>("stable_min_x", stable_min_x, 0.0);
  nh.param<double>("stable_max_y", stable_max_y, 2.0);
  nh.param<double>("stable_min_y", stable_min_y, -2.0);
  nh.param<double>("stable_max_z", stable_max_z, 3.0);
  nh.param<double>("stable_min_z", stable_min_z, 0.0);
  nh.param<double>("min_radius", min_radius, 0.2);
  nh.param<double>("max_radius", max_radius, 1.5);
  nh.param<double>("diff_radius", diff_radius, 0.05);

  nh.param<double>("moving_max_x", moving_max_x, 5.0);
  nh.param<double>("moving_min_x", moving_min_x, 0.0);
  nh.param<double>("moving_max_y", moving_max_y, 2.0);
  nh.param<double>("moving_min_y", moving_min_y, -2.0);
  nh.param<double>("moving_max_z", moving_max_z, 3.0);
  nh.param<double>("moving_min_z", moving_min_z, 0.0);
  nh.param<double>("min_radius", min_radius, 0.2);
  nh.param<double>("max_radius", max_radius, 1.5);
  nh.param<double>("diff_radius", diff_radius, 0.05);


  pt_sub = nh.subscribe("/cloud_registered", 1,
                        &MapGenerate::pointCloudCallback, this);
  marker_pub = nh.advertise<visualization_msgs::Marker>("aim", 1);
}

void MapGenerate::cloudPre(pcl::PointCloud<PointT>::Ptr input_cloud,
                           pcl::PointCloud<PointT>::Ptr output_cloud,
                           double min_x, double max_x, double min_y,
                           double max_y, double min_z, double max_z) {
  // 点云直通滤波
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(input_cloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(min_x, max_x);
  pass.filter(*input_cloud);

  pass.setInputCloud(input_cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(min_y, max_y);
  pass.filter(*input_cloud);

  pass.setInputCloud(input_cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(min_z, max_z);
  pass.filter(*input_cloud);

  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(input_cloud);
  sor.setMeanK(50);
  sor.setStddevMulThresh(1.0);
  sor.filter(*output_cloud);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "map_generate");
  MapGenerate mg;
  ros::spin();
}
