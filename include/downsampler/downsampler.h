#ifndef DOWNSAMPLER_DOWNSAMPLER_H_
#define DOWNSAMPLER_DOWNSAMPLER_H_

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <downsampler/DownsamplerConfig.h>
#include <nodelet/nodelet.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <ecl/geometry.hpp>
#include <tf/LinearMath/Vector3.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <tuple>

namespace downsampler
{

class Downsampler : public nodelet::Nodelet
{
public:
  virtual void onInit();
  virtual void downsample_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);

protected:
  enum PlaneType
  {
    Ground, Ramp, NotATraversablePlane
  };

  virtual void reconfigureCB(DownsamplerConfig &config, uint32_t level);
  virtual pcl::PointCloud<pcl::PointXYZ>::Ptr extractPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  virtual std::vector<float> extractGroundPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  virtual double lowestAngle(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  virtual std::tuple<pcl::PointIndices::Ptr, pcl::PointIndices::Ptr> getGroundIndicies(
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<float> plane);
  virtual std::shared_ptr<std::vector<pcl::PointXYZ> > getPaddingPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                        pcl::PointIndices::Ptr padding_indices);

  virtual pcl::PointCloud<pcl::PointXYZ>::Ptr doStuff(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
                                                      Eigen::Vector3f& axis,
                                                      std::vector<pcl::PointXYZ>& ground_padding_points);
  virtual pcl::PointIndices::Ptr extractRamp(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, Eigen::Vector3f& axis,
                                             boost::shared_ptr<pcl::ModelCoefficients> coeff_out);

  virtual Eigen::Vector3f getRotatedCoeff(std::vector<float>, double degree);
  virtual PlaneType checkPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                               std::vector<pcl::PointXYZ>& ground_padding_points, pcl::PointIndices::Ptr plane_inliers,
                               boost::shared_ptr<pcl::ModelCoefficients> plane_coeff);
  virtual bool checkCommon(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<pcl::PointXYZ>& ground_padding_points,
                           pcl::PointIndices::Ptr plane_inliers);
  virtual nav_msgs::OdometryPtr coeffToOdom(std::vector<float> coeff, std::string name);
  virtual void filterByNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointIndices::Ptr plane_inliers,
                               boost::shared_ptr<pcl::ModelCoefficients> plane_coeff);
  void setGroundNormal();

  void pushBackLastCoeff(std::vector<float>& three_coeffs, std::vector<float> plane_point);

  ros::Subscriber sub_cloud_;
  ros::Publisher pub_downsampled_;
  ros::Publisher pub_filtered_;
  ros::Publisher pub_ground_;
  ros::Publisher pub_padding_;
  ros::Publisher pub_ramp_;
  ros::Publisher pub_result_;
  ros::Publisher pub_ground_pose_;
  ros::Publisher pub_pose_;
  ros::Publisher pub_ramp_pose_;
  ros::Publisher pub_angle_;

  std::shared_ptr<dynamic_reconfigure::Server<DownsamplerConfig> > reconfigure_server_;

  double min_range_;
  double max_range_;
  double leaf_size_;
  double filter_radius_;
  int min_points_threshold_;

  int plane_fitting_type_;
  int plane_max_search_count_;

  int normal_neighbours_;
  double max_angle_error_;

  double plane_max_deviation_;
  double plane_max_degree_;
  double ground_max_degree_;

  double x_rotation_;
  double y_rotation_;
  double z_rotation_;

  std::vector<float> ground_plane_coeff_;
  std::vector<float> sensor_ground_;
  std::vector<float> sensor_ground_downset_;
  std::vector<float> sensor_ground_downset_offset_;
  std::vector<float> sensor_ground_upset_;

  ros::Duration interval_;
  ros::Time next_call_time_;

  tf::TransformBroadcaster transform_broadcaster_;
};

} //end namespace

#endif /* DOWNSAMPLER_DOWNSAMPLER_H_ */
