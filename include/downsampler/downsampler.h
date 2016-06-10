#ifndef DOWNSAMPLER_DOWNSAMPLER_H_
#define DOWNSAMPLER_DOWNSAMPLER_H_

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <downsampler/DownsamplerConfig.h>
#include <nodelet/nodelet.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

namespace downsampler
{

class Downsampler : public nodelet::Nodelet
{
public:
  virtual void onInit();
  virtual void downsample_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);

protected:
  virtual void reconfigureCB(DownsamplerConfig &config, uint32_t level);
  virtual pcl::PointCloud<pcl::PointXYZ>::Ptr extractPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  virtual pcl::PointIndices::Ptr extractRamp(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, std::vector<float>& axis,
                                             boost::shared_ptr<pcl::ModelCoefficients> coeff_out);

  ros::Subscriber sub_cloud_;
  ros::Publisher pub_downsampled_;
  ros::Publisher pub_filtered_;
  ros::Publisher pub_ground_;
  ros::Publisher pub_ramp_;
  ros::Publisher pub_result_;

  std::shared_ptr<dynamic_reconfigure::Server<DownsamplerConfig> > reconfigure_server_;

  double min_range_;
  double max_range_;
  double leaf_size_;
  double filter_radius_;
  int min_points_threshold_;

  int plane_fitting_type_;
  int plane_max_search_count_;
  double plane_max_deviation_;
  double plane_max_angle_;

  ros::Duration interval_;
  ros::Time next_call_time_;

};

} //end namespace

#endif /* DOWNSAMPLER_DOWNSAMPLER_H_ */
