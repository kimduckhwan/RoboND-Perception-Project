#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


DONE = 0
# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    # Bring camaera image (ROS) to PCL to do some CV methods
    pcl_data = ros_to_pcl(pcl_msg)

    # TODO: Voxel Grid Downsampling
    vox = pcl_data.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # Statistical filter
    stat_filter = cloud_filtered.make_statistical_outlier_filter()
    stat_filter.set_mean_k(5)
    x = 0.1
    stat_filter.set_std_dev_mul_thresh(x)
    cloud_stat_filtered = stat_filter.filter()


    # TODO: PassThrough Filter

    passthrough = cloud_stat_filtered.make_passthrough_filter()
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.4
    axis_max = 0.4
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    # Need to apply in Y direction as well since Robot try to guess its arms as book ...

    passthrough2 = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough2.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 0.8
    passthrough2.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough2.filter()

    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance=0.01
    seg.set_distance_threshold(max_distance)

    # TODO: Extract inliers and outliers
    inliers, coefficients = seg.segment()
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)


    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()
    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(5000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                white_cloud[indice][1],
                white_cloud[indice][2],
                rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)


    # TODO: Convert PCL data to ROS messages
    ros_cloud_cluster = pcl_to_ros(cluster_cloud)
    cloud_objects = extracted_outliers
    cloud_table   = extracted_inliers
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table   = pcl_to_ros(cloud_table)

    # TODO: Publish ROS messages

    pcl_points_pub.publish(pcl_msg)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cloud_cluster)


    # Exercise-3 TODOs:
    EX3 = 1
    if (EX3==1):
        # Classify the clusters! (loop through each detected cluster one at a time)
        detected_objects_labels = []
        detected_objects = []

        for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
            pcl_cluster = cloud_objects.extract(pts_list)
            # TODO: convert the cluster from pcl to ROS using helper function
            ros_cluster = pcl_to_ros(pcl_cluster)
            # Extract histogram features
            # TODO: complete this step just as is covered in capture_features.py

            sample_cloud = ros_cluster

            chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            #labeled_features.append([feature, model_name])

            # Make the prediction, retrieve the label for the result
            # and add it to detected_objects_labels list
            prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
            label = encoder.inverse_transform(prediction)[0]
            detected_objects_labels.append(label)

            # Publish a label into RViz
            label_pos = list(white_cloud[pts_list[0]])
            label_pos[2] += 0
            object_markers_pub.publish(make_label(label,label_pos, index))

            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = label
            do.cloud = ros_cluster
            detected_objects.append(do)


        rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

        # Publish the list of detected objects
        # This is the output you'll need to complete the upcoming project!
        detected_objects_pub.publish(detected_objects)

        # Publish the list of detected objects

        # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
        # Could add some logic to determine whether or not your object detections are robust
        # before calling pr2_mover()
        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass

# function to load parameters and request PickPlace service
actual_move = 0
def pr2_mover(object_list):



    # TODO: Initialize variables
    test_scene_num = Int32()
    test_scene_num.data = 1

    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    dict_list = []
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list_param = rospy.get_param('/dropbox')
    # TODO: Parse parameters into individual variables



    for found_object in object_list:    # List of objects found from the image

        found_object_label = found_object.label

        print("Object list length :"+str(len(object_list_param)));

        for i, target_object in enumerate(object_list_param):  # list of objects to found
            if found_object_label == object_list_param[i]['name']:
                object_name.data = str(found_object_label)
                points_arr = ros_to_pcl(found_object.cloud).to_array()
                centroid = np.mean(points_arr, axis=0)[:3]

                # pick position
                pick_pose.position.x = np.asscalar(centroid[0])
                pick_pose.position.y = np.asscalar(centroid[1])
                pick_pose.position.z = np.asscalar(centroid[2])

                # drop position
                if (target_object['group'] == "green"):
                    drop_position = dropbox_list_param[1]['position']
                    place_pose.position.x = drop_position[0]
                    place_pose.position.y = drop_position[1]
                    place_pose.position.z = drop_position[2]
                    arm_name.data = str(dropbox_list_param[1]['name'])
                    print("FOUND "+str(i)+"th Objects", str(found_object_label), "  --> Right")
                else:
                    drop_position = dropbox_list_param[0]['position']
                    place_pose.position.x = drop_position[0]
                    place_pose.position.y = drop_position[1]
                    place_pose.position.z = drop_position[2]
                    arm_name.data = str(dropbox_list_param[0]['name'])
                    print("FOUND "+str(i)+"th Objects", str(found_object_label), "  --> Left")


                yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
                dict_list.append(yaml_dict)

                if (actual_move == 1):
                    rospy.wait_for_service('pick_place_routine')
                    try:
                        pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
                        resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
                        print ("Response: ",resp.success)
                    except rospy.ServiceException, e:
                        print "Service call failed: %s"%e

                break

    # TODO: Output your request parameters into output yaml file
    fname = 'output_'+str(test_scene_num.data)+'.yaml'
    print("FOUND ALL, WRITING YAML: "+fname)
    send_to_yaml(fname, dict_list)
    DONE =1

if __name__ == '__main__':
    if (DONE==0):
        # TODO: ROS node initialization
        rospy.init_node('clustering', anonymous=True)

        # TODO: Create Subscribers
        pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

        # TODO: Create Publishers
        pcl_points_pub = rospy.Publisher("/pcl_points", PointCloud2, queue_size=1)
        pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
        pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
        pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)


        object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
        detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
        # TODO: Load Model From disk
        model = pickle.load(open('model.sav', 'rb'))
        clf = model['classifier']
        encoder = LabelEncoder()
        encoder.classes_ = model['classes']
        scaler = model['scaler']
        # Initialize color_list
        get_color_list.color_list = []

        # TODO: Spin while node is not shutdown
        while not rospy.is_shutdown():
            rospy.spin()
