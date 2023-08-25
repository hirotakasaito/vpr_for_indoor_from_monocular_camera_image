#!/usr/bin/env python3
import argparse
import os
import json
import torch
import math
import numpy as np

from glob import iglob 
from tqdm import tqdm
import cv2
import collections
import numpy as np
import tf
import rospy
import rosbag
from geometry_msgs.msg import Vector3
import random

class RosbagHandler:
    def __init__(self, bagfile):
        print('bagfile: ' + bagfile)
        try:
            self.bag = rosbag.Bag(bagfile)
        except Exception as e:
            rospy.logfatal('failed to load bag file:%s', e)
            exit(1)
        TopicTuple = collections.namedtuple("TopicTuple", ["msg_type", "message_count", "connections", "frequency"])
        TypesAndTopicsTuple =  collections.namedtuple("TypesAndTopicsTuple", ["msg_types", "topics"])
        self.info = self.bag.get_type_and_topic_info()
        for topic, topic_info in self.info.topics.items():
            print("======================================================")
            print("topic_name:      " + topic)
            print("topic_msg_type:  " + topic_info.msg_type)
            print("topic_msg_count: " + str(topic_info.message_count))
            print("frequency:       " + str(topic_info.frequency))
        self.start_time = self.bag.get_start_time()
        self.end_time = self.bag.get_end_time()
        print("start time: " + str(self.start_time))
        print("end time:   " + str(self.end_time))

    def read_messages(self, topics=None, start_time=None, end_time=None, hz=None):
        if start_time is None:
            start_time = self.start_time
        if end_time is None:
            end_time = self.end_time
        start_time = rospy.Time.from_seconds(start_time)
        end_time = rospy.Time.from_seconds(end_time)
        data = {}
        topic_names = []
        for topic in topics:
            data[topic] = []
            topic_names.append("/"+topic)
        for topic, msg, time in self.bag.read_messages(topics=topic_names, start_time=start_time, end_time=end_time):
            if hz is not None:
                data[topic[1:]].append([time.to_nsec()/1e9, msg])
            else:
                data[topic[1:]].append(msg)
        if hz is not None:
            data = self.convert_data(data, hz)
        return data

    def get_topic_type(self, topic_name):
        topic_type = None
        for topic, topic_info in self.info.topics.items():
            if str(topic_name) == topic[1:]:
                topic_type = topic_info.msg_type
        return topic_type

    def convert_data(self, data, hz):
        data_ = {}
        start_time = 0
        end_time = np.inf
        idx = {}
        for topic in data.keys():
            start_time = max(start_time, data[topic][0][0])
            end_time = min(end_time, data[topic][-1][0])
            data_[topic] = []
            idx[topic] = 1
        t = start_time
        while(t<end_time):
            for topic in data.keys():
                cnt = 0
                while(data[topic][idx[topic]][0]<t):
                    cnt+=1
                    idx[topic]+=1
                if (data[topic][idx[topic]][0]-t<t-data[topic][idx[topic]-1][0]):
                    data_[topic].append(data[topic][idx[topic]][1])
                else:
                    data_[topic].append(data[topic][idx[topic]-1][1])
            t+=1./hz
        return data_

def arg_parse():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-bd","--bagfile-dir",type=str,default="/share/private/27th/hirotaka_saito/bagfile/sq2/d_kan1/path_2/")
    parser.add_argument("-od","--output-dir" ,type=str, default="/share/private/27th/hirotaka_saito/dataset/sq2/d_kan1/vpr_path_2_10")
    # parser.add_argument("-tr","--threshold-radius", type=float, default=5)
    parser.add_argument("-ni","--num-image", type=int, default=10)
    parser.add_argument("-to","--topic-odom", type=str, default="t_frog/odom")
    parser.add_argument("-ti","--topic-image", type=str, default="camera/color/image_raw/compressed")
    parser.add_argument("-iw","--image-width", type=int, default="224")
    parser.add_argument("-ih","--image-height", type=int, default="224")

    parser.add_argument("-tor","--threshold-odom-radius", type=float, default="1.0")
    parser.add_argument("-toy","--threshold-odom-yaw", type=float, default="0.3")

    parser.add_argument("-t","--test", type=bool, default=True)
    parser.add_argument("-dc","--divide-count", type=int, default=5)
    parser.add_argument("-msc","--max-save-count", type=int, default=100)
    parser.add_argument("--hz", type=int, default=2)

    args = parser.parse_args()
    return args

def convert_CompressedImage(data, height=None, width=None):
    obs = []
    for msg in data:
        img = cv2.imdecode(np.fromstring(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if height is not None and width is not None:
            h,w,c = img.shape
            img = img[0:h, int((w-h)*0.5):w-int((w-h)*0.5), :]
            img = cv2.resize(img, (height, width))
        obs.append(img)
    return obs

def quaternion_to_euler(quaternion):
    e = tf.transformations.euler_from_quaternion((quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    return Vector3(x=e[0], y=e[1], z=e[2])

def get_pose_from_msg(msg):
    yaw = quaternion_to_euler(msg.pose.pose.orientation).z
    pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
    return pose

def convert_Odometry(data):
    pos = []
    for msg in data:
        pose = get_pose_from_msg(msg)
        pos.append(pose)
    return pos

def transform_pose(pose, base_pose):
    x = pose[0] - base_pose[0]
    y = pose[1] - base_pose[1]
    yaw = pose[2] - base_pose[2]
    trans_pose = np.array([ x*np.cos(base_pose[2]) + y*np.sin(base_pose[2]),
                           -x*np.sin(base_pose[2]) + y*np.cos(base_pose[2]),
                           np.arctan2(np.sin(yaw), np.cos(yaw))]) 
    return trans_pose


def main():
    args = arg_parse()
    divide_time = 1.0 / args.hz / args.divide_count

    topics = [args.topic_image, args.topic_odom]

    # gt_0_count = 0
    # gt_1_count = 0
    # max_0_count = 5000
    # max_1_count = 5000
    bar = tqdm(total = args.max_save_count)
    save_count = 0

    if args.test:
        divide = 20
    else:
        divide = 1

    for bagfile_path in iglob(os.path.join(args.bagfile_dir, "*")):
        print("start:" + bagfile_path)
        file_name = os.path.splitext(os.path.basename(bagfile_path))[0]
        out_dir = os.path.join(args.output_dir, file_name)
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir,'args.json'), 'w') as f:
            json.dump(vars(args),f)

        rosbag_handler = RosbagHandler(bagfile_path)

        for dc in range(args.divide_count):
            # if gt_0_count == max_0_count and gt_1_count == max_1_count:         
            #     print("end")
            #     exit()

            t0 = rosbag_handler.start_time + dc * divide_time
            t1 = rosbag_handler.end_time
            dataset = {}
            sample_data = rosbag_handler.read_messages(topics=topics, start_time=t0, end_time=t1, hz=args.hz)
            for topic in sample_data.keys():
                topic_type = rosbag_handler.get_topic_type(topic)
                if topic_type == "sensor_msgs/CompressedImage":
                    print("==== convert compressed image ====")
                    dataset["obs"] = convert_CompressedImage(sample_data[topic], args.image_width, args.image_height)
                elif topic_type == "nav_msgs/Odometry":
                    print("==== convert odometry ====")
                    dataset["pos"] =  convert_Odometry(sample_data[topic])

            print("==== save data as torch tensor ====")
            
            dataset_traj_obs_list = []
            dataset_traj_pose_list = []
            
            prev_pose = [1e3] * 3
            for i in range(int(len(dataset["obs"]))):
                pose = dataset["pos"][i]
                
                rel_pose = transform_pose(pose, prev_pose)
                if abs(np.linalg.norm(rel_pose[:2])) >= args.threshold_odom_radius  or abs(rel_pose[2]) >= args.threshold_odom_yaw:
                    dataset_traj_obs_list.append(dataset["obs"][i])
                    dataset_traj_pose_list.append(dataset["pos"][i])
                    prev_pose = pose

            dataset_traj_obs_list.pop(0) 
            dataset_traj_pose_list.pop(0) 

            for idx in range(int(len(dataset_traj_obs_list)/args.num_image)):
                t0 = idx*args.num_image
                t1 = t0 + args.num_image

                for i in np.linspace(0, 200, divide):
                    i = int(i)
                    if t1+i >= len(dataset_traj_obs_list):
                        continue

                    file_name = str(dc) + str(idx) + str(i) + ".pt"
                    path = os.path.join(out_dir, file_name)
                    
                    pose1 = dataset_traj_pose_list[t1]
                    pose2 = dataset_traj_pose_list[t1+i]
                    anchor_imgs = dataset_traj_obs_list[t0:t1]

                    rel_pose = transform_pose(pose2, pose1)

                    if abs(np.linalg.norm(rel_pose[:2])) >= 10.0:
                        negative_imgs = dataset_traj_obs_list[t0+i:t1+i]
                        positive_imgs = dataset_traj_obs_list[t0+2:t1+2]

                        tensor_anchor_imgs = torch.tensor(np.array(anchor_imgs), dtype=torch.float32)
                        tensor_positive_imgs = torch.tensor(np.array(positive_imgs), dtype=torch.float32)
                        tensor_negative_imgs = torch.tensor(np.array(negative_imgs), dtype=torch.float32)
                        data ={"anchor_imgs": tensor_anchor_imgs, "positive_imgs":tensor_positive_imgs, "negative_imgs": tensor_negative_imgs}
                    
                        with open(path, "wb") as f:
                            torch.save(data, f)
                            bar.update(1)
                            save_count += 1
                            if save_count >= args.max_save_count:
                                exit()

        print("end:" + bagfile_path)

if __name__ == "__main__":
    main()

