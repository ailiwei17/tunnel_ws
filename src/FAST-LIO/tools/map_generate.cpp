#include <cmath>
#include <fstream>
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

#include <optimer.hpp>
#include <utils.hpp>

typedef pcl::PointXYZ PointT;

class MapGenerate {
public:
  bool debug;
  bool optimal;

  std::string filename;

  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
  pcl::ExtractIndices<PointT> extract;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::search::KdTree<PointT>::Ptr tree;

  pcl::PointCloud<PointT>::Ptr accumulated_cloud;

  pcl::PointCloud<PointT>::Ptr stable_cloud_filtered, moving_cloud_filtered;
  pcl::PointCloud<pcl::Normal>::Ptr stable_cloud_normals, moving_cloud_normals;
  pcl::PointCloud<PointT>::Ptr stable_cloud_filtered2, moving_cloud_filtered2;
  pcl::PointCloud<pcl::Normal>::Ptr stable_cloud_normals2,
      moving_cloud_normals2;
  pcl::ModelCoefficients::Ptr stable_coefficients_plane,
      stable_coefficients_cylinder, moving_coefficients_plane,
      moving_coefficients_cylinder;
  pcl::PointIndices::Ptr stable_inliers_plane, stable_inliers_cylinder,
      moving_inliers_plane, moving_inliers_cylinder;

  pcl::PointCloud<PointT>::Ptr stable_cloud_plane, stable_cloud_cylinder,
      moving_cloud_plane, moving_cloud_cylinder;

  visualization_msgs::Marker stable_bbox_marker, moving_bbox_marker;
  visualization_msgs::Marker stable_midline_marker, moving_midline_marker;

  ros::NodeHandle nh;
  // 直通
  double stable_min_x, stable_max_x, stable_min_y, stable_max_y, stable_min_z,
      stable_max_z;
  double moving_min_x, moving_max_x, moving_min_y, moving_max_y, moving_min_z,
      moving_max_z;

  // 拟合
  double cylinder_normal_weight;
  double cylinder_distance_threshold;

  double plane_normal_weight;
  double plane_distance_threshold;

  ros::Subscriber pt_sub;
  ros::Publisher marker_pub;

  const int window_size = 15;

  // 滑动窗口
  std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> point_cloud_queue;
  int optimal_deque_size;
  optimer::MeanFilter x_mean;
  optimer::MeanFilter y_mean;
  optimer::MeanFilter z_mean;

  // 拟合参数
  double min_radius;
  double max_radius;

  // 半径膨胀
  double diff_radius;

  MapGenerate();
  void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);
    point_cloud_queue.push_back(cloud); // 将新的点云添加到队列末尾

    // 如果队列大小超过了滑动窗口的大小，移除最老的点云
    if (point_cloud_queue.size() > optimal_deque_size) {
      point_cloud_queue.pop_front();
    }

    // 当接收到十次点云后执行操作
    if (point_cloud_queue.size() >= optimal_deque_size) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);
      // 将滑动窗口内的点云叠加起来
      for (const auto &cloud_ptr : point_cloud_queue) {
        *accumulated_cloud += *cloud_ptr;
        std::cout << "point size:" << cloud_ptr->size() << std::endl;
      }
      // 固定圆柱
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
      stable_bbox_marker.color.a = 0.5; // Set the transparency

      stable_midline_marker.header.frame_id =
          msg->header.frame_id; // 设置坐标系
      stable_midline_marker.header.stamp = msg->header.stamp;
      stable_midline_marker.action = visualization_msgs::Marker::ADD;
      stable_midline_marker.scale.x = 0.01; // 线的宽度
      stable_midline_marker.id = 1;
      stable_midline_marker.type = visualization_msgs::Marker::CYLINDER;
      // Set the color (RGBA) of the bounding box
      stable_midline_marker.color.r = 1.0;
      stable_midline_marker.color.g = 0.0;
      stable_midline_marker.color.b = 0.0;
      stable_midline_marker.color.a = 1.0; // Set the transparency

      // 移动
      moving_bbox_marker.header.frame_id = msg->header.frame_id; // 设置坐标系
      moving_bbox_marker.header.stamp = msg->header.stamp;
      moving_bbox_marker.action = visualization_msgs::Marker::ADD;
      moving_bbox_marker.scale.x = 0.01; // 线的宽度
      moving_bbox_marker.id = 2;
      moving_bbox_marker.type = visualization_msgs::Marker::CYLINDER;
      // Set the color (RGBA) of the bounding box
      moving_bbox_marker.color.r = 0.0;
      moving_bbox_marker.color.g = 0.0;
      moving_bbox_marker.color.b = 1.0;
      moving_bbox_marker.color.a = 0.5; // Set the transparency

      moving_midline_marker.header.frame_id =
          msg->header.frame_id; // 设置坐标系
      moving_midline_marker.header.stamp = msg->header.stamp;
      moving_midline_marker.action = visualization_msgs::Marker::ADD;
      moving_midline_marker.scale.x = 0.01; // 线的宽度
      moving_midline_marker.id = 3;
      moving_midline_marker.type = visualization_msgs::Marker::CYLINDER;
      // Set the color (RGBA) of the bounding box
      moving_midline_marker.color.r = 0.0;
      moving_midline_marker.color.g = 0.0;
      moving_midline_marker.color.b = 1.0;
      moving_midline_marker.color.a = 1.0; // Set the transparency

      visualization_msgs::Marker x_distance_marker =
          setMarker(msg->header.frame_id, msg->header.stamp, "x", 10, "x");
      visualization_msgs::Marker y_distance_marker =
          setMarker(msg->header.frame_id, msg->header.stamp, "y", 11, "y");
      visualization_msgs::Marker z_distance_marker =
          setMarker(msg->header.frame_id, msg->header.stamp, "z", 12, "z");

      cloudPre(accumulated_cloud, stable_cloud_filtered, stable_min_x,
               stable_max_x, stable_min_y, stable_max_y, stable_min_z,
               stable_max_z);
      normals_estimate(tree, stable_cloud_filtered, stable_cloud_normals);
      plane_seg(stable_cloud_filtered, stable_cloud_normals,
                stable_inliers_plane, stable_coefficients_plane,
                plane_normal_weight, plane_distance_threshold);
      get_plane(stable_cloud_filtered, stable_inliers_plane,
                stable_cloud_plane);
      remove_plane(stable_cloud_normals, stable_inliers_plane,
                   stable_cloud_filtered2, stable_cloud_normals2);
      cylinder_seg(stable_cloud_filtered2, stable_cloud_normals2,
                   stable_inliers_cylinder, stable_coefficients_cylinder,
                   cylinder_normal_weight, cylinder_distance_threshold);
      get_cylinder(stable_cloud_filtered, stable_cloud_filtered2,
                   stable_inliers_cylinder, stable_cloud_cylinder);

      cloudPre(accumulated_cloud, moving_cloud_filtered, moving_min_x,
               moving_max_x, moving_min_y, moving_max_y, moving_min_z,
               moving_max_z);
      normals_estimate(tree, moving_cloud_filtered, moving_cloud_normals);
      plane_seg(moving_cloud_filtered, moving_cloud_normals,
                moving_inliers_plane, moving_coefficients_plane,
                plane_normal_weight, plane_distance_threshold);
      get_plane(moving_cloud_filtered, moving_inliers_plane,
                moving_cloud_plane);
      remove_plane(moving_cloud_normals, moving_inliers_plane,
                   moving_cloud_filtered2, moving_cloud_normals2);
      cylinder_seg(moving_cloud_filtered2, moving_cloud_normals2,
                   moving_inliers_cylinder, moving_coefficients_cylinder,
                   cylinder_normal_weight, cylinder_distance_threshold);
      get_cylinder(moving_cloud_filtered, moving_cloud_filtered2,
                   moving_inliers_cylinder, moving_cloud_cylinder);

      generateCylinderPointCloud(
          10, 0.001, 8, diff_radius, stable_coefficients_cylinder,
          stable_cloud_cylinder, stable_bbox_marker, stable_midline_marker);

      generateCylinderPointCloud(
          10, 0.001, 8, diff_radius, moving_coefficients_cylinder,
          moving_cloud_cylinder, moving_bbox_marker, moving_midline_marker);

      std::string x_output =
          savePartNum(100 * (stable_bbox_marker.pose.position.x -
                             moving_bbox_marker.pose.position.x));
      std::string y_output =
          savePartNum(100 * (std::abs((stable_bbox_marker.pose.position.y -
                                       moving_bbox_marker.pose.position.y)) -
                             0.5 * std::abs(stable_bbox_marker.scale.z +
                                            moving_bbox_marker.scale.z)));
      std::string z_output =
          savePartNum(100 * (stable_bbox_marker.pose.position.z -
                             moving_bbox_marker.pose.position.z));
      if(optimal){
        x_mean.push(std::stod(x_output));
        y_mean.push(std::stod(y_output));
        z_mean.push(std::stod(z_output));
        x_output = std::to_string(x_mean.get());
        y_output = std::to_string(y_mean.get());
        z_output = std::to_string(z_mean.get());
      }
      x_distance_marker.text =
          "X :  " + x_output + " cm"; // Set the text you want to display
      marker_pub.publish(x_distance_marker);
      y_distance_marker.text =
          "Y* :  " + y_output + " cm"; // Set the text you want to display
      marker_pub.publish(y_distance_marker);
      z_distance_marker.text =
          "Z :  " + z_output + " cm"; // Set the text you want to display
      marker_pub.publish(z_distance_marker);

      std::cout << debug << std::endl;

      if (debug) {
        // Save data to CSV
        utils::FileManager::saveToCSV(filename, x_output, y_output, z_output);
      }

      // 重置计数器和累积点云
      accumulated_cloud->clear();
    }
  }
  visualization_msgs::Marker setMarker(std::string frame_id, ros::Time ts,
                                       std::string ns, int id,
                                       std::string axis) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    ; // Set the appropriate frame ID
    marker.header.stamp = ts;

    marker.ns = "ns";
    marker.id = id;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = 0.0;
    marker.pose.position.y = 0.0;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 0.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;

    marker.scale.x = 1.0; // Set the text scale
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;

    marker.color.r = 0.0; // Set text color (red in this example)
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.color.a = 1.0; // Set alpha (1.0 is fully opaque)

    if (axis == "x") {
      marker.pose.position.z = 2.0;
      marker.color.r = 1.0;
    }
    if (axis == "y") {
      marker.pose.position.z = 3.0;
      marker.color.g = 1.0;
    }
    if (axis == "z") {
      marker.pose.position.z = 4.0;
      marker.color.b = 1.0;
    }

    return marker;
  };
  std::string savePartNum(double num) {
    std::string output = std::to_string(num);
    int decimal_places = 2; // number of decimal places to keep
    size_t decimal_point = output.find('.');
    if (decimal_point != std::string::npos &&
        decimal_point + decimal_places < output.size()) {
      output.erase(decimal_point + decimal_places + 1, std::string::npos);
    }
    return output;
  };
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
                 pcl::ModelCoefficients::Ptr output_coefficients_plane,
                 double plane_normal_weight = 0.1,
                 double plane_distance_threshold = 0.02) {
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(plane_normal_weight);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(plane_distance_threshold); // 0.03
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
                    pcl::ModelCoefficients::Ptr output_coefficients_cylinder,
                    double cylinder_normal_weight = 0.1,
                    double cylinder_distance_threshold = 0.3) {
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(cylinder_normal_weight);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(cylinder_distance_threshold); // 0.05
    seg.setRadiusLimits(min_radius, max_radius);
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
      visualization_msgs::Marker &input_bbox_marker,
      visualization_msgs::Marker &input_midline_marker) {
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

        Eigen::Vector3d axis_vector(input_coefficients_cylinder->values[3],
                                    input_coefficients_cylinder->values[4],
                                    input_coefficients_cylinder->values[5]);
        Eigen::Vector3d up_vector(0.0, 0.0, -1.0);
        Eigen::Vector3d right_vector = axis_vector.cross(up_vector);
        right_vector.normalized();
        Eigen::Quaterniond q(Eigen::AngleAxisd(
            -1.0 * std::acos(axis_vector.dot(up_vector)), right_vector));
        q.normalize();

        input_bbox_marker.pose.orientation.x = q.x();
        input_bbox_marker.pose.orientation.y = q.y();
        input_bbox_marker.pose.orientation.z = q.z();
        input_bbox_marker.pose.orientation.w = q.w();

        // Set the scale of the bounding box
        // 局部坐标系
        input_bbox_marker.scale.x = 2 * r0;
        input_bbox_marker.scale.y = 2 * r0;
        input_bbox_marker.scale.z = std::abs(max_point_OBB.x - min_point_OBB.x);

        marker_pub.publish(input_bbox_marker);

        input_midline_marker.pose.position.x = position_OBB.x;
        input_midline_marker.pose.position.y = position_OBB.y;
        input_midline_marker.pose.position.z = position_OBB.z;
        input_midline_marker.pose.orientation.x = q.x();
        input_midline_marker.pose.orientation.y = q.y();
        input_midline_marker.pose.orientation.z = q.z();
        input_midline_marker.pose.orientation.w = q.w();

        // Set the scale of the bounding box
        // 局部坐标系
        input_midline_marker.scale.x = 0.1;
        input_midline_marker.scale.y = 0.1;
        input_midline_marker.scale.z = 0.2 * long_length;

        marker_pub.publish(input_midline_marker);
      }
    }
  }
};

MapGenerate::MapGenerate() {
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

  nh.param<bool>("debug", debug, false);
  nh.param<bool>("optimal", optimal, false);
  nh.param<int>("optimal_deque_size", optimal_deque_size, 5);
  if (optimal) {
    x_mean.init(optimal_deque_size);
    y_mean.init(optimal_deque_size);
    z_mean.init(optimal_deque_size);
  }

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

  nh.param<double>("plane_normal_weight", plane_normal_weight, 0.1);
  nh.param<double>("plane_distance_threshold", plane_distance_threshold, 0.02);

  nh.param<double>("cylinder_normal_weight", cylinder_normal_weight, 0.1);
  nh.param<double>("cylinder_distance_threshold", cylinder_distance_threshold,
                   0.3);

  filename = "/home/liwei/tunnel_ws/result/output_data.csv";
  utils::FileManager::fileReset(filename);

  pt_sub =
      nh.subscribe("/livox/lidar", 1, &MapGenerate::pointCloudCallback, this);
  marker_pub = nh.advertise<visualization_msgs::Marker>("aim", 5);
}

void MapGenerate::cloudPre(pcl::PointCloud<PointT>::Ptr input_cloud,
                           pcl::PointCloud<PointT>::Ptr output_cloud,
                           double min_x, double max_x, double min_y,
                           double max_y, double min_z, double max_z) {
  // 创建一个新的 PointCloud 对象用于存储滤波后的结果
  pcl::PointCloud<PointT>::Ptr filtered_cloud(new pcl::PointCloud<PointT>);

  // 复制 input_cloud 到 filtered_cloud
  *filtered_cloud = *input_cloud;

  // 点云直通滤波
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(filtered_cloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(min_x, max_x);
  pass.filter(*filtered_cloud);

  pass.setInputCloud(filtered_cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(min_y, max_y);
  pass.filter(*filtered_cloud);

  pass.setInputCloud(filtered_cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(min_z, max_z);
  pass.filter(*filtered_cloud);

  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(filtered_cloud);
  sor.setMeanK(50);
  sor.setStddevMulThresh(1.0);
  sor.filter(*output_cloud); // 将结果存储到 output_cloud
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "map_generate");
  MapGenerate mg;
  ros::spin();
}
