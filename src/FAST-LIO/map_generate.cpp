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

#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/common/concatenate.h>

#include <string>

#include <ros/ros.h>

#include <ceres/ceres.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
typedef pcl::PointXYZ PointT;

// 损失函数
struct CostFun {

  CostFun(double x, double y, double z) : x_(x), y_(y), z_(z) {}

  template <typename T>
  bool operator()(const T *const parm, T *residual) const {
    residual[0] = pow((x_ - parm[0]), 2) + pow((y_ - parm[1]), 2) +
                  pow((z_ - parm[2]), 2) -
                  pow(parm[3] * (x_ - parm[0]) + parm[4] * (y_ - parm[1]) +
                          parm[5] * (z_ - parm[2]),
                      2) -
                  pow(parm[6], 2);
    return true;
  }
  const double x_, y_, z_;
};

class MapGenerate {
public:
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
  pcl::ExtractIndices<PointT> extract;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::search::KdTree<PointT>::Ptr tree;

  pcl::PointCloud<PointT>::Ptr stable_cloud_filtered;
  pcl::PointCloud<pcl::Normal>::Ptr stable_cloud_normals;
  pcl::PointCloud<PointT>::Ptr stable_cloud_filtered2;
  pcl::PointCloud<pcl::Normal>::Ptr stable_cloud_normals2;
  pcl::ModelCoefficients::Ptr stable_coefficients_plane,
      stable_coefficients_cylinder;
  pcl::PointIndices::Ptr stable_inliers_plane, stable_inliers_cylinder;

  pcl::PointCloud<PointT>::Ptr stable_cloud_plane;
  pcl::PointCloud<PointT>::Ptr stable_cloud_cylinder;
  ros::NodeHandle nh;
  // 直通
  double min_x, max_x, min_y, max_y, min_z, max_z;
  // 体素
  float voxel_size_x, voxel_size_y, voxel_size_z;

  MapGenerate();
  void readPcdFile();
  void cloudPre();
  void normals_estimate() {
    ne.setSearchMethod(tree);
    ne.setInputCloud(stable_cloud_filtered);
    ne.setKSearch(50);
    ne.compute(*stable_cloud_normals);
  }

  void plane_seg() {
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setInputCloud(stable_cloud_filtered);
    seg.setInputNormals(stable_cloud_normals);
    seg.segment(*stable_inliers_plane, *stable_coefficients_plane);
    std::cerr << "Plane coefficients: " << *stable_coefficients_plane
              << std::endl;
  }

  void get_plane() {
    extract.setInputCloud(stable_cloud_filtered);
    extract.setIndices(stable_inliers_plane);
    extract.setNegative(false);

    extract.filter(*stable_cloud_plane);
    pcl::PCDWriter writer;
    std::string file_name = std::string("plane.pcd");
    std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") +
                               file_name);
    writer.write<pcl::PointXYZ>(all_points_dir, *cloud, false);
  }

  void remove_plane() {
    extract.setNegative(true);
    extract.filter(*stable_cloud_filtered2);
    extract_normals.setNegative(true);
    extract_normals.setInputCloud(stable_cloud_normals);
    extract_normals.setIndices(stable_inliers_plane);
    extract_normals.filter(*stable_cloud_normals2);
  }

  void cylinder_seg() {
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(0.05);
    seg.setRadiusLimits(0.0, 1.5);
    seg.setInputCloud(stable_cloud_filtered2);
    seg.setInputNormals(stable_cloud_normals2);

    // 圆柱体轴线在X轴方向上的位置
    // 圆柱体轴线在Y轴方向上的位置
    // 圆柱体轴线在Z轴方向上的位置
    // 表示圆柱体在X轴方向上的朝向（单位向量）
    // 表示圆柱体在Y轴方向上的朝向（单位向量）
    // 表示圆柱体在Z轴方向上的朝向（单位向量）
    // 圆柱体的半径

    seg.segment(*stable_inliers_cylinder, *stable_coefficients_cylinder);
    std::cerr << "Cylinder coefficients: " << *stable_coefficients_cylinder
              << std::endl;
  }

  void get_cylinder() {
    extract.setInputCloud(stable_cloud_filtered2);
    extract.setIndices(stable_inliers_cylinder);
    extract.setNegative(false);

    extract.filter(*stable_cloud_cylinder);
    if (stable_cloud_cylinder->points.empty())
      std::cerr << "Can't find the cylindrical component." << std::endl;
    else {
      std::cerr << "PointCloud representing the cylindrical component: "
                << stable_cloud_cylinder->points.size() << " data points."
                << std::endl;
      pcl::PCDWriter writer;
      std::string file_name = std::string("cylinder.pcd");
      std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") +
                                 file_name);
      writer.write<pcl::PointXYZ>(all_points_dir, *stable_cloud_cylinder,
                                  false);
    }
  }

  void ceres_solve() {
    //设置优化初值
    double x0 = stable_coefficients_cylinder->values[0];
    double y0 = stable_coefficients_cylinder->values[1];
    double z0 = stable_coefficients_cylinder->values[2];
    double a = stable_coefficients_cylinder->values[3];
    double b = stable_coefficients_cylinder->values[4];
    double c = stable_coefficients_cylinder->values[5];
    double r0 = stable_coefficients_cylinder->values[6];

    //开始优化
    //设置参数块
    double param[7] = {x0, y0, z0, a, b, c, r0};
    //定义优化问题
    ceres::Problem problem;
    for (int i = 0; i < stable_cloud_cylinder->size(); i++) {
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<CostFun, 1, 7>(
              new CostFun(stable_cloud_cylinder->points[i].x,
                          stable_cloud_cylinder->points[i].y,
                          stable_cloud_cylinder->points[i].z)),
          nullptr, param);
    }
    //配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; //增量方程求解
    options.minimizer_progress_to_stdout = true;               //输出到cout
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "优化之后的结果为:" << std::endl;
    for (auto x : param) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
    std::cout << "优化后半径为" << param[6] << std::endl;

    // 更新圆柱的参数方程
    stable_coefficients_cylinder->values[0] = param[0];
    stable_coefficients_cylinder->values[1] = param[1];
    stable_coefficients_cylinder->values[2] = param[2];
    stable_coefficients_cylinder->values[3] = param[3];
    stable_coefficients_cylinder->values[4] = param[4];
    stable_coefficients_cylinder->values[5] = param[5];
    stable_coefficients_cylinder->values[6] = param[6];

    //计算优化后的参数误差
    double r_ = param[6];
    double sum_error = 0;
    std::vector<double> error;
    double ave_error = 0;
    for (int i = 0; i < stable_cloud_cylinder->size(); i++) {
      double temp1 = pow(stable_cloud_cylinder->points[i].x - param[0], 2) +
                     pow(stable_cloud_cylinder->points[i].y - param[1], 2) +
                     pow(stable_cloud_cylinder->points[i].z - param[2], 2);
      double temp2 =
          pow(param[3] * (stable_cloud_cylinder->points[i].x - param[0]) +
                  param[4] * (stable_cloud_cylinder->points[i].y - param[1]) +
                  param[5] * (stable_cloud_cylinder->points[i].z - param[2]),
              2);
      double r_predict_2 = temp1 - temp2;
      double r_predict = sqrt(r_predict_2);
      double temp_error = r_ - r_predict;
      double temp_error_2 = pow(temp_error, 2);
      sum_error += temp_error_2;
      error.push_back(temp_error);
    }
    ave_error = sqrt(sum_error / stable_cloud_cylinder->size());
    std::cout << "平均误差为" << ave_error << std::endl;

    //绘制图形
    std::vector<double> index;
    for (int i = 0; i < stable_cloud_cylinder->size(); i++) {
      index.push_back(i);
    }
  }

  void generateCylinderPointCloud(const double z_resolution,
                                  const double theta_num) {
    // 先构建轴线为Z轴的圆柱面点云
    int Num = theta_num;
    float inter = 2.0 * M_PI / Num;
    Eigen::RowVectorXd vectorx(Num), vectory(Num), vectorz(Num);
    Eigen::RowVector3d axis(stable_coefficients_cylinder->values[3],
                            stable_coefficients_cylinder->values[4],
                            stable_coefficients_cylinder->values[5]);
    float length = axis.norm();
    vectorx.setLinSpaced(Num, 0, Num - 1);
    vectory = vectorx;
    float x0, y0, z0, r0;
    x0 = stable_coefficients_cylinder->values[0];
    y0 = stable_coefficients_cylinder->values[1];
    z0 = stable_coefficients_cylinder->values[2];
    r0 = stable_coefficients_cylinder->values[6];

    pcl::PointCloud<pcl::PointXYZN>::Ptr raw(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr final(
        new pcl::PointCloud<pcl::PointXYZ>);

    for (float z(-10); z <= 10; z += z_resolution) {
      for (auto i = 0; i < Num; ++i) {
        pcl::PointXYZ point;
        point.x = r0 * cos(vectorx[i] * inter);
        point.y = r0 * sin(vectory[i] * inter);
        point.z = z;
        raw->points.push_back(point);
      }
    }

    raw->width = (int)raw->size();
    raw->height = 1;
    raw->is_dense = false;

    // 点云旋转 Z轴转到axis
    Eigen::RowVector3d Z(0.0, 0.0, 0.1), T0(0, 0, 0),
        T(stable_coefficients_cylinder->values[0],
          stable_coefficients_cylinder->values[1],
          stable_coefficients_cylinder->values[2]);
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
    pcl::getMinMax3D(*stable_cloud_cylinder, min_pt, max_pt);
    Eigen::Vector3f cylinder_axis(
        stable_coefficients_cylinder->values[3],
        stable_coefficients_cylinder->values[4],
        stable_coefficients_cylinder->values[5]); // 圆柱体轴线方向向量
    // 保存为txt
    std::string pcd_name = std::string("generate_cylinder.pcd");
    std::string pcd_dir(std::string(std::string(ROOT_DIR) + "PCD/") + pcd_name);
    std::string txt_name = std::string("points.txt");
    std::string txt_dir =
        std::string(std::string(ROOT_DIR) + "PCD/") + txt_name;

    std::ofstream ofs(txt_dir);
    for (size_t i = 0; i < tmp->points.size(); ++i) {
      pcl::PointXYZ point = tmp->points[i];
      if (point.y < max_pt.y && point.y > min_pt.y) {
        final->push_back(point);
        ofs << point.x << " " << point.y << " " << point.z << std::endl;
        // TODO: 如何根据点和圆柱方程计算法向量，并转换为a,b,c
      }
    }
    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZ>(pcd_dir, *final, false);
    //--------------------------------------可视化--------------------------
    pcl::visualization::PCLVisualizer viewer;
    viewer.addPointCloud<pcl::PointXYZ>(final, "cloud");
    viewer.addPointCloud<pcl::PointXYZ>(stable_cloud_cylinder, "cloud2");
    viewer.addCoordinateSystem();

    while (!viewer.wasStopped()) {
      viewer.spinOnce(100);
    }
  }
};

MapGenerate::MapGenerate() {
  cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
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
  nh.param<double>("max_x", max_x, 5.0);
  nh.param<double>("min_x", min_x, 0.0);
  nh.param<double>("max_y", max_y, 2.0);
  nh.param<double>("min_y", min_y, -2.0);
  nh.param<double>("max_z", max_z, 3.0);
  nh.param<double>("min_z", min_z, 0.0);
  nh.param<float>("voxel_size_x", voxel_size_x, 0.2);
  nh.param<float>("voxel_size_y", voxel_size_y, 0.5);
  nh.param<float>("voxel_size_z", voxel_size_z, 0.2);
}

void MapGenerate::readPcdFile() {
  std::string file_name = std::string("scans.pcd");
  std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") +
                             file_name);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(all_points_dir, *cloud) ==
      -1) //* load the file
  {
    PCL_ERROR("Couldn't read file pcd \n");
  }
  std::cout << "Loaded " << cloud->width * cloud->height << std::endl;
}

void MapGenerate::cloudPre() {
  // 点云直通滤波
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("x");
  pass.setFilterLimits(min_x, max_x);
  pass.filter(*cloud);

  pass.setInputCloud(cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(min_y, max_y);
  pass.filter(*cloud);

  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(min_z, max_z);
  pass.filter(*cloud);

  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(50);
  sor.setStddevMulThresh(1.0);
  sor.filter(*stable_cloud_filtered);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "map_generate");
  MapGenerate mg;
  mg.readPcdFile();
  mg.cloudPre();
  mg.normals_estimate();
  // 分割平面
  mg.plane_seg();
  // 保存平面
  mg.get_plane();
  // 移除平面
  mg.remove_plane();
  // 圆柱分割
  mg.cylinder_seg();
  // 获取圆柱
  mg.get_cylinder();
  // 优化圆柱
  mg.ceres_solve();
  // 生成轨迹
  mg.generateCylinderPointCloud(0.05, 6);
}
