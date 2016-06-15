#include <downsampler/downsampler.h>
#include <pluginlib/class_list_macros.h>

#include <tf/transform_listener.h>

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
  pub_padded_ground_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("padded_ground_points", 1);
  pub_result_ = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("result_points", 1);

  pub_pose_ = nh.advertise<nav_msgs::Odometry>("plane_pose", 50);
  pub_ground_pose_ = nh.advertise<nav_msgs::Odometry>("ground_pose", 50);
  pub_ramp_pose_ = nh.advertise<nav_msgs::Odometry>("ramp_pose", 50);

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
  private_nh.param("plane_max_search_count", plane_max_search_count_, 150);
  private_nh.param("plane_max_deviation", plane_max_deviation_, 0.02);
  private_nh.param("plane_max_angle_degree", plane_max_angle_, 5.0);
  plane_max_angle_ = DEG2RAD(plane_max_angle_);

  private_nh.param("normal_neighbours", normal_neighbours_, 8);
  private_nh.param("max_angle_error_", max_angle_error_, 25.0);

  double rate;
  private_nh.param("rate", rate, 30.0);

  if (rate == 0)
    interval_ = ros::Duration(0);
  else
    interval_ = ros::Duration(1.0 / rate);
  next_call_time_ = ros::Time::now();

  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  ROS_INFO_STREAM("Downsampling points using a leaf size of '" << leaf_size_ << "' m, running at " << rate << " Hz.");

  lookedup_ = false;
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
  plane_max_angle_ = config.plane_max_angle_degree;
  plane_max_angle_ = DEG2RAD(plane_max_angle_);

  normal_neighbours_ = config.normal_neighbours;
  max_angle_error_ = config.max_angle_error;
}

void Downsampler::downsample_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  if (!lookedup_)
  {
    tf::TransformListener tf_listener;
    tf::StampedTransform tf_footprint_to_sensor;
    std::string sensor_frame = "sensor_3d_short_range_depth_optical_frame"; ///TODO get automatically

    tf_listener.waitForTransform("base_footprint", sensor_frame, ros::Time::now(), ros::Duration(1.0));
    tf_listener.lookupTransform("base_footprint", sensor_frame, ros::Time(0), tf_footprint_to_sensor);
    tf_listener.lookupTransform("base_footprint", sensor_frame, ros::Time(0), tf_footprint_to_sensor);

    tf::Vector3 z_axis_point(0.0, 0.0, 0.1); ///TODO maybe different point; take care of vector direction
    robot_axis_in_camera_frame_ = tf_footprint_to_sensor(z_axis_point);
    robot_axis_in_camera_frame_.normalize();
    robot_center_in_camera_frame_ = tf_footprint_to_sensor(tf::Vector3(0, 0, 0));
    robot_center_in_camera_frame_.normalize();

    ROS_ERROR_STREAM(
        "robot_axis_in_camera_frame_: " << robot_axis_in_camera_frame_.getX() << ", " << robot_axis_in_camera_frame_.getY() << ", " << robot_axis_in_camera_frame_.getZ());
    ROS_ERROR_STREAM(
        "robot_center_in_camera_frame_: " << robot_center_in_camera_frame_.getX() << ", " << robot_center_in_camera_frame_.getY() << ", " << robot_center_in_camera_frame_.getZ());

    lookedup_ = true;
  }

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
  ROS_WARN("------------------------------------");

  pcl::PointCloud<pcl::PointXYZ>::Ptr ground_free(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>());

  boost::shared_ptr<pcl::ModelCoefficients> ground_coefficients(new pcl::ModelCoefficients);

  pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices());

  pcl::SACSegmentation<pcl::PointXYZ> segmentation;

  segmentation.setOptimizeCoefficients(true);
  segmentation.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  segmentation.setMethodType(plane_fitting_type_);
  segmentation.setDistanceThreshold(plane_max_deviation_ + 0.02); //we gotta catch'em all, so set that low
  segmentation.setMaxIterations(plane_max_search_count_); //wanna be the very best, so set that high

  segmentation.setAxis(Eigen::Vector3f(0, -0.095, -0.226)); ///TODO set automatically
  segmentation.setEpsAngle(plane_max_angle_);

  segmentation.setInputCloud(cloud);
  segmentation.segment(*ground_inliers, *ground_coefficients);

  ///TODO make sure the ground is near the ground

  if (pub_padded_ground_.getNumSubscribers() > 0)
  {
    pcl::ExtractIndices<pcl::PointXYZ> extract_ground;
    extract_ground.setInputCloud(cloud);
    extract_ground.setIndices(ground_inliers);
    extract_ground.filter(*ground);

    sensor_msgs::PointCloud2Ptr ground_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*ground, *ground_msg);
    pub_padded_ground_.publish(ground_msg);
  }

  std::vector<pcl::PointXYZ> ground_buffer_points = filterIndices(cloud, ground_coefficients, ground_inliers);

  pcl::ExtractIndices<pcl::PointXYZ> remove_ground;
  remove_ground.setInputCloud(cloud);
  remove_ground.setIndices(ground_inliers);
  remove_ground.setNegative(true);
  remove_ground.filter(*ground_free);

  if (pub_ground_.getNumSubscribers() > 0)
  {
    ground->clear();

    pcl::ExtractIndices<pcl::PointXYZ> extract_ground;
    extract_ground.setInputCloud(cloud);
    extract_ground.setIndices(ground_inliers);
    extract_ground.filter(*ground);

    sensor_msgs::PointCloud2Ptr ground_msg(new sensor_msgs::PointCloud2);
    pcl::toROSMsg(*ground, *ground_msg);
    pub_ground_.publish(ground_msg);
  }

  if (ground_inliers->indices.size() == 0 || ground_coefficients->values.size() == 0)
  {
    ROS_INFO_THROTTLE(10, "[Downsampler]: Could not extract ground plane, no ground in field of view?");
    return cloud;
  }

  double ground_d = ground_coefficients->values[3];

  if (ground_d < 0)
  {
    for (int i = 0; i < ground_coefficients->values.size(); ++i)
    {
      ground_coefficients->values[i] *= -1;
    }
  }

  Eigen::Vector3f rotated_up = getRotatedNormal(ground_coefficients, 3.0);
  Eigen::Vector3f rotated_down = getRotatedNormal(ground_coefficients, -3.0);

  result = doStuff(ground_free, rotated_up, ground_buffer_points);

  if(result->empty())
  {
    ROS_INFO("----------------");
    result = doStuff(ground_free, rotated_down, ground_buffer_points);
  }

  if(result->empty())
  {
    return ground_free;
  }

  return result;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Downsampler::doStuff(pcl::PointCloud<pcl::PointXYZ>::Ptr ground_free,
                                                         Eigen::Vector3f& axis,
                                                         std::vector<pcl::PointXYZ>& ground_buffer_points)
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

    //      ros::Duration(5.0).sleep();

    if (plane_coefficients->values.size() == 0)
    {
      ROS_INFO("[Downsampler]: Could not extract ramp, no ramp in field of view?");
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

    if (checkPlane(cloud, ground_buffer_points, plane_inliers, plane_coefficients) == Ramp)
    {
      pub_ramp_pose_.publish(coeffToOdom(plane_coefficients, "ramp"));

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

std::vector<pcl::PointXYZ> Downsampler::filterIndices(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                                      boost::shared_ptr<pcl::ModelCoefficients> plane_coeff,
                                                      pcl::PointIndices::Ptr indices)
{
  std::vector<int> kept_indices;
  std::vector<pcl::PointXYZ> removed_points;

  std::vector<int>::iterator it;

  for (it = indices->indices.begin(); it != indices->indices.end(); ++it)
  {
    if (pcl::pointToPlaneDistance(cloud->at(*it), plane_coeff->values[0], plane_coeff->values[1],
                                  plane_coeff->values[2], plane_coeff->values[3]) <= 0.02)
    {
      kept_indices.push_back(*it);
    }
    else
    {
      removed_points.push_back(cloud->at(*it));
    }
  }

  indices->indices.clear();
  indices->indices.insert(indices->indices.begin(), kept_indices.begin(), kept_indices.end());

  ROS_WARN_STREAM("kept_indices: " << indices->indices.size() << ", removed_points: " << removed_points.size());

  return removed_points;
}

Eigen::Vector3f Downsampler::getRotatedNormal(boost::shared_ptr<pcl::ModelCoefficients> coeff, double degree)
{
  double a = coeff->values[0];
  double b = coeff->values[1];
  double c = coeff->values[2];
  double d = coeff->values[3];

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
  segmentation.setEpsAngle(DEG2RAD(2)); //5 degree = 0.0872665 ///TODO

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  segmentation.setInputCloud(input_cloud);
  segmentation.segment(*inliers, *coeff_out);

  return inliers;
}

Downsampler::PlaneType Downsampler::checkPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                               std::vector<pcl::PointXYZ>& ground_buffer_points,
                                               pcl::PointIndices::Ptr plane_inliers,
                                               boost::shared_ptr<pcl::ModelCoefficients> plane_coeff)
{
  pub_pose_.publish(coeffToOdom(plane_coeff, "plane"));

  double plane_to_sensor_distance = plane_coeff->values[3];

  ROS_INFO_STREAM("plane_to_sensor_distance: " << plane_to_sensor_distance);

  std::vector<int>::iterator it_inlier;
  std::vector<pcl::PointXYZ>::iterator it_ground;

  int count_common = 0;

  for (it_inlier = plane_inliers->indices.begin(); it_inlier != plane_inliers->indices.end(); ++it_inlier)
  {
    pcl::PointXYZ& point = cloud->at(*it_inlier);

    for (it_ground = ground_buffer_points.begin(); it_ground != ground_buffer_points.end(); ++it_ground)
    {
      if (it_ground->x == point.x && it_ground->y == point.y && it_ground->y == point.y)
      {
        ++count_common;
      }
    }
  }

  double min_indicies = std::min(ground_buffer_points.size(), plane_inliers->indices.size());

  double min_common_ratio = 0.1;
  double common_ratio = count_common / min_indicies;

  ROS_INFO_STREAM(
      "common_ratio: " << common_ratio << "; ground_buffer_points size: " << ground_buffer_points.size() << ", plane_inliers size: " << plane_inliers->indices.size());

  if (common_ratio < min_common_ratio)
  {
    ROS_ERROR_STREAM("Common ratio < " << min_common_ratio << ": " << common_ratio << "; count: " << count_common);
    return NotATraversablePlane;
  }
  else
  {
    ROS_INFO_STREAM("Common ratio < " << min_common_ratio << ": " << common_ratio << "; count: " << count_common);
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::ExtractIndices<pcl::PointXYZ> extract_plane;
  extract_plane.setInputCloud(cloud);
  extract_plane.setIndices(plane_inliers);
  extract_plane.filter(*plane);

  //it's a seemingly traversable slope but maybe it is just random objects which look like that
  //check if some random normals fit
  if (normalsOfPointsSupportThePlane(plane, plane_coeff))
  {
    return Ramp;
  }

  return NotATraversablePlane;
}

nav_msgs::OdometryPtr Downsampler::coeffToOdom(boost::shared_ptr<pcl::ModelCoefficients> coeff, std::string name)
{
  double a = coeff->values[0];
  double b = coeff->values[1];
  double c = coeff->values[2];
  double d = coeff->values[3];

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

bool Downsampler::normalsOfPointsSupportThePlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
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

  cloud_it = cloud->begin();

  double error_sum = 0.0;
  double error = 0.0;

  int erase_count = 0;

  for (it = cloud_normals->begin(); it != cloud_normals->end(); ++it)
  {
    Eigen::Vector3f normal = (*it).getNormalVector3fMap();
    error = std::acos(plane_normal.dot(normal));

    if (std::isnan(error))
    {
      error = 0;
    }

    if (std::abs(error) >= DEG2RAD(90))
    {
      error = error - std::copysign(DEG2RAD(180), error);
    }

    error = std::abs(error);

    if (error > DEG2RAD(max_angle_error_))
    {
      ++erase_count;
//      cloud_it = cloud->erase(cloud_it);
    }
    else
    {
      ++cloud_it;
    }

//    ROS_WARN_STREAM("Error: " << error);
    error_sum += error;
  }

  error_sum = error_sum / cloud_normals->size();

  ROS_INFO_STREAM("cloud_normals size: " << cloud_normals->size());
  ROS_INFO_STREAM("erase_count size: " << erase_count);

  double max_error = DEG2RAD(max_angle_error_);

  if (error_sum >= max_error)
  {
    ROS_ERROR_STREAM(
        "angle_error > " << RAD2DEG(max_error) << " deg: " << RAD2DEG(error_sum) << "; size: " << cloud_normals->size());
    return false;
  }

  ROS_WARN_STREAM("angle_error: " << RAD2DEG(error_sum) << "; size: " << cloud_normals->size());

  return true;
}

bool Downsampler::approximateNormal(pcl::Normal normal_out)
{
  return true;
}

} //end namespace

PLUGINLIB_EXPORT_CLASS(downsampler::Downsampler, nodelet::Nodelet)
