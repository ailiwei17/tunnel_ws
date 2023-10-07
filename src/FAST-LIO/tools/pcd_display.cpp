#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl_conversions//pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    ros::init(argc,argv,"UanBdet");
    ros::NodeHandle nh;
    ros::Publisher pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("pcl_output",1);
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    sensor_msgs::PointCloud2 output;
    std::string file_name;
    nh.param<std::string>("file_name",file_name,"/home/liwei/tunnel_ws/src/FAST-LIO/PCD/scans.pcd");

    pcl::io::loadPCDFile(file_name,cloud);
    pcl::toROSMsg(cloud,output);
    output.header.frame_id = "world";   
    ros::Rate loop_rate(1);
    while(ros::ok())
    {
        pcl_pub.publish(output);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}

