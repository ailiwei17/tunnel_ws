/**
 * @file navigation_model.h
 * @author liwei
 * @brief  两点间巡航/原地自旋
 * @version 0.1
 * @date 2023-02-27
 *
 *
 */
#ifndef NAVIGATION_MODEL_H
#define NAVIGATION_MODEL_H
 
#include <actionlib/client/simple_action_client.h>
#include <geometry_msgs/PoseStamped.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <ros/ros.h>
#include <vector>
//给MoveBaseAction定义一个别名，方便创建对象
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> Client;
 
class GoalSending {
 private:
  // ros
  ros::NodeHandle nh_;
  ros::Publisher goal_pub_;
  ros::Timer timer_;
  geometry_msgs::PoseStamped target_pose_;
  move_base_msgs::MoveBaseGoal goal_;
  //创建MoveBaseAction对象
  Client ac_;
 
  bool initialized_;
  int count = 0;

  std::vector<double> first_goal;
  std::vector<double> second_goal;
 
  /**
   * @brief 发布目标点
   */
  void goalPointPub(const ros::TimerEvent& event);
  /**
   * @brief 读取目标点
   */
  void goalPointRead();
  //判断是否接收到目标点
  void activeCb();
  void doneCb(const actionlib::SimpleClientGoalState& state,
              const move_base_msgs::MoveBaseResultConstPtr& result);
 
 public:
  GoalSending();
};
 
#endif  // NAVIGATION_MODEL_H