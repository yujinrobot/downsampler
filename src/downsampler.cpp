#include <downsampler/downsampler.h>
#include <pluginlib/class_list_macros.h>

#include <tf/transform_listener.h>
#include <limits>

#include <boost/shared_ptr.hpp>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/console/print.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <std_msgs/Float32.h>

#include <algorithm>

namespace downsampler
{

void Downsampler::onInit()
{
  ros::NodeHandle& nh = getNodeHandle();

  sub_cloud_ = nh.subscribe<sensor_msgs::PointCloud2>("input_cloud", 1, &Downsampler::downsample_cloud_cb, this);
  pub_downsampled_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("downsampled_points", 1);
  pub_filtered_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("filtered_points", 1);
  pub_ramp_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("ramp_points", 1);
  pub_ground_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("ground_points", 1);
  pub_padding_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("padding_points", 1);
  pub_result_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("result_points", 1);

  pub_pose_ = nh.advertise<nav_msgs::Odometry>("plane_pose", 50);
  pub_ground_pose_ = nh.advertise<nav_msgs::Odometry>("ground_pose", 50);
  pub_ramp_pose_ = nh.advertise<nav_msgs::Odometry>("ramp_pose", 50);

  pub_angle_ = nh.advertise<std_msgs::Float32>("angle", 50);

  reconfigure_server_ = std::shared_ptr<dynamic_reconfigure::Server<DownsamplerConfig> >(
      new dynamic_reconfigure::Server<DownsamplerConfig>(nh));
  dynamic_reconfigure::Server<DownsamplerConfig>::CallbackType reconfigure_cb = boost::bind(&Downsampler::reconfigureCB,
                                                                                            this, _1, _2);
  reconfigure_server_->setCallback(reconfigure_cb);

  ros::NodeHandle& private_nh = getPrivateNodeHandle();

  private_nh.param("min_range", min_range_, 0.0);
  private_nh.param("max_range", max_range_, 4.0);
  private_nh.param("leaf_size", leaf_size_, 0.03);
  private_nh.param("filter_radius", filter_radius_, 0.03);
  private_nh.param("min_points_threshold", min_points_threshold_, 3);

  private_nh.param("plane_fitting_type", plane_fitting_type_, pcl::SAC_LMEDS);
  private_nh.param("plane_max_search_count", plane_max_search_count_, 120);
  private_nh.param("plane_max_deviation", plane_max_deviation_, 0.02);
  private_nh.param("plane_max_angle_degree", plane_max_degree_, 6.0);

  private_nh.param("ground_max_angle", ground_max_degree_, 3.0);

  private_nh.param("normal_neighbours", normal_neighbours_, 8);
  private_nh.param("max_angle_error_", max_angle_error_, 20.0);

  double rate;
  private_nh.param("rate", rate, 30.0);

  if (rate == 0)
    interval_ = ros::Duration(0);
  else
    interval_ = ros::Duration(1.0 / rate);
  next_call_time_ = ros::Time::now();

  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  ROS_INFO_STREAM("Downsampling points using a leaf size of '" << leaf_size_ << "' m, running at " << rate << " Hz.");
}

void Downsampler::reconfigureCB(DownsamplerConfig &config, uint32_t level)
{
  min_range_ = config.min_range;
  max_range_ = config.max_range;
  leaf_size_ = config.leaf_size;
  filter_radius_ = config.filter_radius;
  min_points_threshold_ = config.min_points_threshold;

  plane_fitting_type_ = config.plane_fitting_type;
  plane_max_search_count_ = config.plane_max_search_count;
  plane_max_deviation_ = config.plane_max_deviation;
  plane_max_degree_ = config.plane_max_degree;
  ground_max_degree_ = config.ground_max_degree;

  normal_neighbours_ = config.normal_neighbours;
  max_angle_error_ = config.max_angle_error;

  x_rotation_ = config.x_rotation;
  y_rotation_ = config.y_rotation;
  z_rotation_ = config.z_rotation;
}

void Downsampler::downsample_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  if (ros::Time::now() <= next_call_time_)
    return;
  next_call_time_ = next_call_time_ + interval_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cut_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::PointCloud<pcl::PointXYZ> input_cloud;
  pcl::fromROSMsg(*cloud_msg, input_cloud);

  if (max_range_ != 0.0)
  {
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(input_cloud.makeShared());
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_range_, max_range_);
    pass.filter(*cut_cloud);
  }
  else
  {
    cut_cloud = input_cloud.makeShared();
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if (leaf_size_ == 0.0)
  {
    downsampled_cloud = cut_cloud;
  }
  else
  {
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cut_cloud);
    sor.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    sor.filter(*downsampled_cloud);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if (min_points_threshold_ > 0 && filter_radius_ > 0.0)
  {
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outlier_removal;
    outlier_removal.setInputCloud(downsampled_cloud);
    outlier_removal.setRadiusSearch(filter_radius_);
    outlier_removal.setMinNeighborsInRadius(min_points_threshold_);
    outlier_removal.filter(*filtered_cloud);
  }
  else
  {
    filtered_cloud = downsampled_cloud;
  }

  if (pub_downsampled_.getNumSubscribers() > 0)
  {
    sensor_msgs::PointCloud2Ptr result_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*filtered_cloud, *result_msg);

    pub_downsampled_.publish(result_msg);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr rampless_cloud = extractPlanes(filtered_cloud);

  sensor_msgs::PointCloud2Ptr result_msg(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(*rampless_cloud, *result_msg);

  pub_result_.publish(result_msg);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Downsampler::extractPlanes(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
//  ROS_WARN("------------------------------------");

  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_free(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>());

//  if(ground_plane_coeff_.empty())
  {
    setGroundNormal();
  }

  std::vector<float> estimated_ground_coeff = extractGroundPlane(cloud);
  std::tuple<pcl::PointIndices::Ptr, pcl::PointIndices::Ptr> ground_inliers = getGroundIndicies(cloud,
                                                                                                estimated_ground_coeff);
  pcl::PointIndices::Ptr ground_indices = std::get<0>(ground_inliers);
  pcl::PointIndices::Ptr padding_indices = std::get<1>(ground_inliers);

  Eigen::Vector3f normal(estimated_ground_coeff[0], estimated_ground_coeff[1], estimated_ground_coeff[2]);

  std::shared_ptr<std::vector<pcl::PointXYZ> > padding_points = getPaddingPoints(cloud, padding_indices);

  if (pub_padding_.getNumSubscribers() > 0)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr padding(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ExtractIndices<pcl::PointXYZ> extract_padding;
    extract_padding.setInputCloud(cloud);
    extract_padding.setIndices(padding_indices);
    extract_padding.filter(*padding);

    sensor_msgs::PointCloud2Ptr ground_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*padding, *ground_msg);
    pub_padding_.publish(ground_msg);
  }

  pcl::ExtractIndices<pcl::PointXYZ> remove_ground;
  remove_ground.setInputCloud(cloud);
  remove_ground.setIndices(ground_indices);
  remove_ground.setNegative(true);
  remove_ground.filter(*ground_free);

  if (pub_ground_.getNumSubscribers() > 0)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::ExtractIndices<pcl::PointXYZ> extract_ground;
    extract_ground.setInputCloud(cloud);
    extract_ground.setIndices(ground_indices);
    extract_ground.filter(*ground);

    sensor_msgs::PointCloud2Ptr ground_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*ground, *ground_msg);
    pub_ground_.publish(ground_msg);
  }

//  Eigen::Vector3f rotated_up = getRotatedNormal(ground_coefficients, 3.0);
//  Eigen::Vector3f rotated_down = getRotatedNormal(ground_coefficients, -3.0);

  result = doStuff(ground_free, normal, *padding_points);

//  if(result->empty())
//  {
//    ROS_INFO("----------------");
//    result = doStuff(ground_free, rotated_down, ground_buffer_points);
//  }

  if (result->empty())
  {
    return ground_free;
  }

  return result;
}

std::vector<float> Downsampler::extractGroundPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  double lowest_angle = lowestAngle(cloud);

  Eigen::Vector3f rotated_ground_normal = getRotatedCoeff(ground_plane_coeff_, lowest_angle);

  std::vector<float> lower_plane_coeff;
  lower_plane_coeff.push_back(rotated_ground_normal(0));
  lower_plane_coeff.push_back(rotated_ground_normal(1));
  lower_plane_coeff.push_back(rotated_ground_normal(2));
  pushBackLastCoeff(lower_plane_coeff, sensor_ground_);

  coeffToOdom(lower_plane_coeff, "lower");

  return lower_plane_coeff;
}

double Downsampler::lowestAngle(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  Eigen::Vector3f ground_downset_point(sensor_ground_downset_[0], sensor_ground_downset_[1], sensor_ground_downset_[2]);
  Eigen::Vector3f ground_downset_point_offset(sensor_ground_downset_offset_[0], sensor_ground_downset_offset_[1],
                                              sensor_ground_downset_offset_[2]);

  ///TODO only do once
  tf::Transform transform_down;
  transform_down.setOrigin(tf::Vector3(ground_downset_point(0), ground_downset_point(1), ground_downset_point(2)));
  transform_down.setRotation(tf::Quaternion(0, 0, 0, 1));
  transform_broadcaster_.sendTransform(
      tf::StampedTransform(transform_down, ros::Time::now(), "sensor_3d_short_range_depth_optical_frame", "down"));

  tf::Transform transform_down_off;
  transform_down_off.setOrigin(
      tf::Vector3(ground_downset_point_offset(0), ground_downset_point_offset(1), ground_downset_point_offset(2)));
  transform_down_off.setRotation(tf::Quaternion(0, 0, 0, 1));
  transform_broadcaster_.sendTransform(
      tf::StampedTransform(transform_down_off, ros::Time::now(), "sensor_3d_short_range_depth_optical_frame", "off"));

  Eigen::Vector3f ground_normal(ground_plane_coeff_[0], ground_plane_coeff_[1], ground_plane_coeff_[2]);
  ground_normal.normalize();

  Eigen::Vector3f sub = ground_downset_point_offset - ground_downset_point;
  Eigen::Vector3f sub2;
  Eigen::Vector3f normal;

  double lowest_angle = DEG2RAD(plane_max_degree_);
  double angle = 0;

  Eigen::Vector3f point;
  pcl::PointXYZ lowest_point;

  ///TODO slice the point cloud up to meaningful possible points to not iterate of every point
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud->begin(); it != cloud->end(); ++it)
  {
    point = Eigen::Vector3f(it->x, it->y, it->z);
    sub2 = point - ground_downset_point;

    normal = sub2.cross(sub);
    normal.normalize();

    angle = std::atan2(normal(2), normal(1)) - std::atan2(ground_normal(2), ground_normal(1));
//    angle = std::acos(normal.dot(ground_normal));

    if (std::isnan(angle))
    {
      angle = 0.0;
    }

//    if (std::abs(angle) >= DEG2RAD(90))
//    {
//      angle = angle - std::copysign(DEG2RAD(180), angle);
//    }

//    ROS_INFO_STREAM("normal: " << normal);
//    ROS_INFO_STREAM("ground_normal: " << ground_normal);
//    ROS_INFO_STREAM("angle: " << RAD2DEG(angle));

//    double a = coeff[0];
//    double b = coeff[1];
//    double c = coeff[2];
//
//    Eigen::Vector3f normal(a, b, c);
//  //  normal.normalize();
//
//    Eigen::Vector3f axis(1, 0, 0);
//    Eigen::AngleAxis<float> rotation(DEG2RAD(degree), axis);
//
//    Eigen::Vector3f rotated_normal = rotation.toRotationMatrix() * normal;

    if (std::abs(angle) <= DEG2RAD(plane_max_degree_) && angle < lowest_angle)
    {
      lowest_angle = angle;
      lowest_point = *it;
    }
  }

  std_msgs::Float32 msgs;
  msgs.data = RAD2DEG(lowest_angle);
  pub_angle_.publish(msgs);

//  ROS_INFO_STREAM(
//      "lowest point: " << lowest_point.x << ", " << lowest_point.y << ", " << lowest_point.z << "; lowest_angle: " << RAD2DEG(lowest_angle));

  tf::Transform transform;
  transform.setOrigin(tf::Vector3(lowest_point.x, lowest_point.y, lowest_point.z));
  transform.setRotation(tf::Quaternion(0, 0, 0, 1));
  transform_broadcaster_.sendTransform(
      tf::StampedTransform(transform, ros::Time::now(), "sensor_3d_short_range_depth_optical_frame", "lowest")); ///TODO

  return lowest_angle;
}

std::tuple<pcl::PointIndices::Ptr, pcl::PointIndices::Ptr> Downsampler::getGroundIndicies(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<float> plane)
{
  pcl::PointIndices::Ptr indicies(new pcl::PointIndices());
  pcl::PointIndices::Ptr indicies_padded(new pcl::PointIndices());

  double distance = 0;

  for (int i = 0; i < cloud->points.size(); ++i)
  {
    distance = pcl::pointToPlaneDistanceSigned(cloud->points[i], plane[0], plane[1], plane[2], plane[3]);

    if (distance <= 0.03) ///TODO
    {
      if (distance <= 0.02)
      {
        indicies->indices.push_back(i);
//        if (distance < 0.0)
//        {
//          //point under the ground -> remove because is noise
//        }
      }
      else
      {
        indicies_padded->indices.push_back(i);
      }
    }
  }

  return std::tuple<pcl::PointIndices::Ptr, pcl::PointIndices::Ptr>(indicies, indicies_padded);
}

std::shared_ptr<std::vector<pcl::PointXYZ> > Downsampler::getPaddingPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                                           pcl::PointIndices::Ptr padding_indices)
{
  std::shared_ptr<std::vector<pcl::PointXYZ> > padding_points(new std::vector<pcl::PointXYZ>());

  for (std::vector<int>::iterator it = padding_indices->indices.begin(); it != padding_indices->indices.end(); ++it)
  {
    padding_points->push_back(cloud->points[*it]);
  }

  return padding_points;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Downsampler::doStuff(pcl::PointCloud<pcl::PointXYZ>::Ptr ground_free,
                                                         Eigen::Vector3f& axis,
                                                         std::vector<pcl::PointXYZ>& ground_padding_points)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>(*ground_free));
  pcl::PointCloud<pcl::PointXYZ>::Ptr ramp(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr removed_planes(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>());

  boost::shared_ptr<pcl::ModelCoefficients> plane_coefficients(new pcl::ModelCoefficients);

  int i = 0;
  bool found = false;
  while (i <= 3)
  {
    pcl::PointIndices::Ptr plane_inliers = extractRamp(cloud, axis, plane_coefficients);
    ++i;

    if (plane_coefficients->values.size() == 0)
    {
//      ROS_INFO("[Downsampler]: Could not extract ramp, no ramp in field of view?");
      break;
    }

    if (pub_ramp_.getNumSubscribers() > 0)
    {
      pcl::ExtractIndices<pcl::PointXYZ> extract_ramp;
      extract_ramp.setInputCloud(cloud);
      extract_ramp.setIndices(plane_inliers);
      extract_ramp.filter(*ramp);

      sensor_msgs::PointCloud2Ptr plane_msg(new sensor_msgs::PointCloud2);
      pcl::toROSMsg(*ramp, *plane_msg);
      pub_ramp_.publish(plane_msg);
    }

//    ros::Duration(5.0).sleep();

    if (checkPlane(cloud, ground_padding_points, plane_inliers, plane_coefficients) == Ramp)
    {
//      pub_ramp_pose_.publish(coeffToOdom(plane_coefficients, "ramp"));

      result->clear();

      pcl::ExtractIndices<pcl::PointXYZ> extract_result;
      extract_result.setInputCloud(cloud);
      extract_result.setNegative(true);
      extract_result.setIndices(plane_inliers);
      extract_result.filter(*result);

      //move any non-ramp planes back in
      *result = *result + *removed_planes;

      found = true;
      break;
    }

    //wrong plane extracted
    //remove it but keep it in the result
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::ExtractIndices<pcl::PointXYZ> extract_plane;
    extract_plane.setInputCloud(cloud);
    extract_plane.setIndices(plane_inliers);
    extract_plane.filter(*plane);

    *removed_planes = *removed_planes + *plane;

    pcl::PointCloud<pcl::PointXYZ>::Ptr new_ground_free(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::ExtractIndices<pcl::PointXYZ> extract_plane_invert;
    extract_plane.setInputCloud(cloud);
    extract_plane.setIndices(plane_inliers);
    extract_plane.setNegative(true);
    extract_plane.filter(*new_ground_free);

    cloud = new_ground_free;
  }

  if (pub_ramp_.getNumSubscribers() > 0)
  {
    if (!found)
      ramp->clear();

    sensor_msgs::PointCloud2Ptr plane_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*ramp, *plane_msg);
    pub_ramp_.publish(plane_msg);
  }

  if (!found)
    result->clear();

  return result;
}

Eigen::Vector3f Downsampler::getRotatedCoeff(std::vector<float> coeff, double degree)
{
//rotated around the point defined by the "normal of the vector";

  double a = coeff[0];
  double b = coeff[1];
  double c = coeff[2];

  Eigen::Vector3f normal(a, b, c);
//  normal.normalize();

  Eigen::Vector3f axis(1, 0, 0);
  Eigen::AngleAxis<float> rotation(DEG2RAD(degree), axis);

  Eigen::Vector3f rotated_normal = rotation.toRotationMatrix() * normal;

  tf::Vector3 tf_vec(rotated_normal(0), rotated_normal(1), rotated_normal(2));

  tf::Quaternion q(0, 0, 0, 1);

  tf::Transform transform;
  transform.setOrigin(tf_vec);
  transform.setRotation(q);
  transform_broadcaster_.sendTransform(
      tf::StampedTransform(transform, ros::Time::now(), "sensor_3d_short_range_depth_optical_frame", "rotated_normal"));

  return rotated_normal;
}

pcl::PointIndices::Ptr Downsampler::extractRamp(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, Eigen::Vector3f& axis,
                                                boost::shared_ptr<pcl::ModelCoefficients> coeff_out)
{
  pcl::SACSegmentation<pcl::PointXYZ> segmentation;

  segmentation.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  segmentation.setOptimizeCoefficients(true);
  segmentation.setMethodType(plane_fitting_type_);
  segmentation.setMaxIterations(plane_max_search_count_);
  segmentation.setDistanceThreshold(plane_max_deviation_);

  segmentation.setAxis(Eigen::Vector3f(axis(0), axis(1), axis(2)));
  segmentation.setEpsAngle(DEG2RAD(plane_max_degree_)); //5 degree = 0.0872665 ///TODO

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  segmentation.setInputCloud(input_cloud);
  segmentation.segment(*inliers, *coeff_out);

  return inliers;
}

Downsampler::PlaneType Downsampler::checkPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                               std::vector<pcl::PointXYZ>& ground_padding_points,
                                               pcl::PointIndices::Ptr plane_inliers,
                                               boost::shared_ptr<pcl::ModelCoefficients> plane_coeff)
{
//  pub_pose_.publish(coeffToOdom(plane_coeff, "plane"));

  if (!ground_padding_points.empty() && !checkCommon(cloud, ground_padding_points, plane_inliers))
  {
    return NotATraversablePlane;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::ExtractIndices<pcl::PointXYZ> extract_plane;
  extract_plane.setInputCloud(cloud);
  extract_plane.setIndices(plane_inliers);
  extract_plane.filter(*plane);

  filterByNormals(plane, plane_inliers, plane_coeff);

  if (plane_inliers->indices.empty())
  {
    return NotATraversablePlane;
  }

  if (!ground_padding_points.empty() && checkCommon(cloud, ground_padding_points, plane_inliers))
  {
    return Ramp;
  }

  return NotATraversablePlane;
}

bool Downsampler::checkCommon(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                              std::vector<pcl::PointXYZ>& ground_padding_points, pcl::PointIndices::Ptr plane_inliers)
{
  std::vector<int>::iterator it_inlier;
  std::vector<pcl::PointXYZ>::iterator it_ground;

  int count_common = 0;

  for (it_inlier = plane_inliers->indices.begin(); it_inlier != plane_inliers->indices.end(); ++it_inlier)
  {
    pcl::PointXYZ& point = cloud->at(*it_inlier);

    for (it_ground = ground_padding_points.begin(); it_ground != ground_padding_points.end(); ++it_ground)
    {
      if (it_ground->x == point.x && it_ground->y == point.y && it_ground->y == point.y)
      {
        ++count_common;
      }
    }
  }

  double min_indicies = std::min(ground_padding_points.size(), plane_inliers->indices.size());

  double min_common_ratio = 0.1;
  double common_ratio = count_common / min_indicies;

//  ROS_INFO_STREAM(
//      "ground_buffer_points size: " << ground_buffer_points.size() << ", plane_inliers size: " << plane_inliers->indices.size());

  if (common_ratio < min_common_ratio)
  {
//    ROS_ERROR_STREAM("Common ratio < " << min_common_ratio << ": " << common_ratio << "; count: " << count_common);
    return false;
  }
  else
  {
//    ROS_INFO_STREAM("Common ratio < " << min_common_ratio << ": " << common_ratio << "; count: " << count_common);
  }

  return true;
}

nav_msgs::OdometryPtr Downsampler::coeffToOdom(std::vector<float> coeff, std::string name)
{
  double a = coeff[0];
  double b = coeff[1];
  double c = coeff[2];
  double d = coeff[3];

  tf::Vector3 normal(a, b, c);
  normal.normalize();

  double length_squared = a * a + b * b + c * c;

//get closest point to the origin of the plane
  double scale = -d / length_squared;
  double x = a * scale;
  double y = b * scale;
  double z = c * scale;

  tf::Vector3 axis(1, 0, 0);
  tf::Vector3 rotation_axis = axis.cross(normal);
  double rotation_angle = std::acos(axis.dot(normal) / (normal.length() * axis.length()));

  tf::Quaternion q;
  q.setRotation(rotation_axis, rotation_angle);

  tf::Transform transform;
  transform.setOrigin(tf::Vector3(x, y, z));
  transform.setRotation(q);
  transform_broadcaster_.sendTransform(
      tf::StampedTransform(transform, ros::Time::now(), "sensor_3d_short_range_depth_optical_frame", name)); ///TODO

  nav_msgs::OdometryPtr odom_msg = nav_msgs::OdometryPtr(new nav_msgs::Odometry());
  odom_msg->header.stamp = ros::Time::now();
  odom_msg->header.frame_id = "sensor_3d_short_range_depth_optical_frame"; ///TODO

  geometry_msgs::Quaternion odom_quat;
  tf::quaternionTFToMsg(q, odom_quat);

  odom_msg->pose.pose.position.x = x;
  odom_msg->pose.pose.position.y = y;
  odom_msg->pose.pose.position.z = z;
  odom_msg->pose.pose.orientation = odom_quat;

  odom_msg->child_frame_id = name;

  return odom_msg;
}

void Downsampler::filterByNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointIndices::Ptr plane_inliers,
                                  boost::shared_ptr<pcl::ModelCoefficients> plane_coeff)
{
  Eigen::Vector3f plane_normal(plane_coeff->values[0], plane_coeff->values[1], plane_coeff->values[2]);

  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;

  normal_estimation.setSearchMethod(tree);
  normal_estimation.setInputCloud(cloud);
  normal_estimation.setKSearch(normal_neighbours_);
  normal_estimation.compute(*cloud_normals);

///TODO
//make an array of n = ? random indices
//and check if k % of the normals fit to plane

  pcl::PointCloud<pcl::Normal>::iterator it;
  pcl::PointCloud<pcl::PointXYZ>::iterator cloud_it;
  std::vector<int> kept_inliers;

  cloud_it = cloud->begin();

  double error = 0.0;

  int inlier_index = 0;

  for (it = cloud_normals->begin(); it != cloud_normals->end(); ++it, ++inlier_index)
  {
    Eigen::Vector3f normal = (*it).getNormalVector3fMap();
    error = std::acos(plane_normal.dot(normal));

    if (std::isnan(error))
    {
      continue;
    }

    if (std::abs(error) >= DEG2RAD(90))
    {
      error = error - std::copysign(DEG2RAD(180), error);
    }

    if (std::abs(error) <= DEG2RAD(max_angle_error_))
    {
      kept_inliers.push_back(plane_inliers->indices[inlier_index]);
    }
  }

//  ROS_INFO_STREAM("kept " << kept_inliers.size() << " of " << plane_inliers->indices.size());

  plane_inliers->indices.clear();
  plane_inliers->indices.insert(plane_inliers->indices.begin(), kept_inliers.begin(), kept_inliers.end());
}

void Downsampler::setGroundNormal()
{
  tf::TransformListener tf_listener;
  tf::StampedTransform tf_footprint_to_sensor;
  tf::StampedTransform tf_sensor_to_footprint;
  std::string sensor_frame = "sensor_3d_short_range_depth_optical_frame"; ///TODO get automatically

  tf_listener.waitForTransform(sensor_frame, "base_footprint", ros::Time::now(), ros::Duration(1.0));
  tf_listener.lookupTransform(sensor_frame, "base_footprint", ros::Time(0), tf_footprint_to_sensor);
  tf_listener.lookupTransform("base_footprint", sensor_frame, ros::Time(0), tf_sensor_to_footprint);

//  tf::Quaternion adjustment;
//  adjustment.setRPY(x_rotation_, z_rotation_, y_rotation_);
//  tf::Quaternion before = tf_footprint_to_sensor.getRotation();
//  ROS_INFO_STREAM("before: " << before.x() << ", " << before.y() << ", " << before.z() << ", " << before.w());
//  tf_footprint_to_sensor.setRotation(tf_footprint_to_sensor.getRotation() * adjustment);
//  tf::Quaternion after = tf_footprint_to_sensor.getRotation();
//  ROS_INFO_STREAM("after: " << after.x() << ", " << after.y() << ", " << after.z() << ", " << after.w());
//  ground_normal = tf::quatRotate(adjustment, ground_normal);

  tf::Vector3 origin = tf_sensor_to_footprint.getOrigin();

  tf::Quaternion q(0, 0, 0, 1);

  tf::Vector3 sensor_ground_point_in_sensor = tf_footprint_to_sensor(tf::Vector3(origin.getX(), 0.0, 0.0));
  tf::Vector3 sensor_ground_downset_in_sensor = tf_footprint_to_sensor(
      tf::Vector3(origin.getX(), 0.0, -plane_max_deviation_));
  tf::Vector3 sensor_ground_downset_offset_in_sensor = tf_footprint_to_sensor(
      tf::Vector3(origin.getX(), 1.0, -plane_max_deviation_));
  tf::Vector3 sensor_ground_upset_in_sensor = tf_footprint_to_sensor(
      tf::Vector3(origin.getX(), 0.0, plane_max_deviation_));

  tf::Vector3 add_point_in_footprint(0.0, 0.0, origin.getZ()); ///TODO maybe different point; take care of vector direction
  tf::Vector3 footprint_in_sensor = tf_footprint_to_sensor(tf::Vector3(0, 0, 0));
  tf::Vector3 add_point_in_sensor = tf_footprint_to_sensor(add_point_in_footprint);

  sensor_ground_.push_back(sensor_ground_point_in_sensor.getX());
  sensor_ground_.push_back(sensor_ground_point_in_sensor.getY());
  sensor_ground_.push_back(sensor_ground_point_in_sensor.getZ());

  sensor_ground_downset_.push_back(sensor_ground_downset_in_sensor.getX());
  sensor_ground_downset_.push_back(sensor_ground_downset_in_sensor.getY());
  sensor_ground_downset_.push_back(sensor_ground_downset_in_sensor.getZ());

  sensor_ground_downset_offset_.push_back(sensor_ground_downset_offset_in_sensor.getX());
  sensor_ground_downset_offset_.push_back(sensor_ground_downset_offset_in_sensor.getY());
  sensor_ground_downset_offset_.push_back(sensor_ground_downset_offset_in_sensor.getZ());

  sensor_ground_upset_.push_back(sensor_ground_upset_in_sensor.getX());
  sensor_ground_upset_.push_back(sensor_ground_upset_in_sensor.getY());
  sensor_ground_upset_.push_back(sensor_ground_upset_in_sensor.getZ());

  tf::Transform transform_result;
  transform_result.setOrigin(sensor_ground_point_in_sensor);
  transform_result.setRotation(tf_footprint_to_sensor.getRotation());

  tf::StampedTransform tf_sensor_to_plane = tf::StampedTransform(transform_result, ros::Time::now(),
                                                                 "sensor_3d_short_range_depth_optical_frame",
                                                                 "sensor_ground");
  transform_broadcaster_.sendTransform(tf_sensor_to_plane);

  tf::Vector3 ground_normal = add_point_in_sensor - footprint_in_sensor;

//  ROS_INFO_STREAM("test: " << ground_normal.getX() << ", " << ground_normal.getY() << ", " << ground_normal.getZ());

  tf::Transform transform_test;
  transform_test.setOrigin(ground_normal);
  transform_test.setRotation(q);
  transform_broadcaster_.sendTransform(
      tf::StampedTransform(transform_test, ros::Time::now(), "sensor_3d_short_range_depth_optical_frame",
                           "ground_normal"));

//  ground_normal = tf::quatRotate(adjustment, ground_normal);

  ground_plane_coeff_.push_back(ground_normal.getX());
  ground_plane_coeff_.push_back(ground_normal.getY());
  ground_plane_coeff_.push_back(ground_normal.getZ());
  ground_plane_coeff_.push_back(origin.getZ());
}

void Downsampler::pushBackLastCoeff(std::vector<float>& three_coeffs, std::vector<float> plane_point)
{
//ax + by + cz + d = 0
//-d = ax + by + cz

  double d = three_coeffs[0] * plane_point[0] + three_coeffs[1] * plane_point[1] + three_coeffs[2] * plane_point[2];

  three_coeffs.push_back(-d);
}

} //end namespace

PLUGINLIB_EXPORT_CLASS(downsampler::Downsampler, nodelet::Nodelet)
