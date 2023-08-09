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

    parser.add_argument("-bd","--bagfile-dir",type=str,default="/share/private/27th/hirotaka_saito/bagfile/sq2/d_kan1/path/")
    parser.add_argument("-od","--output-dir" ,type=str, default="/share/private/27th/hirotaka_saito/dataset/sq2/d_kan1/vpr_test/")
    parser.add_argument("-tr","--threshold-radius", type=float, default=2)
    parser.add_argument("-ni","--num-image", type=int, default=5)
    parser.add_argument("-to","--topic-odom", type=str, default="t_frog/odom")
    parser.add_argument("-ti","--topic-image", type=str, default="camera/color/image_raw/compressed")
    parser.add_argument("-iw","--image-width", type=int, default="224")
    parser.add_argument("-ih","--image-height", type=int, default="224")
    parser.add_argument("-t","--test", type=bool, default=False)
    parser.add_argument("-dc","--divide-count", type=int, default=1)
    parser.add_argument("--hz", type=int, default=0.5)

    args = parser.parse_args()
    return args

def convert_CompressedImage(data, height=None, width=None):
    obs = []
    for msg in tqdm(data):
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
    for msg in tqdm(data):
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

    gt_0_count = 0
    gt_1_count = 0
    max_0_count = 5000
    max_1_count = 5000

    for bagfile_path in iglob(os.path.join(args.bagfile_dir, "*")):
        print("start:" + bagfile_path)
        file_name = os.path.splitext(os.path.basename(bagfile_path))[0]
        out_dir = os.path.join(args.output_dir, file_name)
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir,'args.json'), 'w') as f:
            json.dump(vars(args),f)

        rosbag_handler = RosbagHandler(bagfile_path)

        for dc in tqdm(range(args.divide_count)):
            if gt_0_count == max_0_count and gt_1_count == max_1_count:         
                print("end")
                exit()

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
            for idx in tqdm(range(int(len(dataset["obs"])/args.num_image))):
                t0 = idx*args.num_image
                t1 = t0+args.num_image
                for i in range(50):
                    i = int(i)
                    file_name = str(dc) + str(idx) + str(i) + ".pt"
                    path = os.path.join(out_dir, file_name)

                    if (t0+i+args.num_image) >= len(dataset["pos"]):
                        continue

                    pose1 = dataset["pos"][t1]
                    pose2 = dataset["pos"][t0+i+args.num_image]
                    rel_pose = transform_pose(pose2, pose1)
                    dis = math.sqrt(rel_pose[0] * rel_pose[0] + rel_pose[1] * rel_pose[1])

                    if dis > args.threshold_radius:
                        gt = 0.0
                        if gt_0_count >= max_0_count:
                            continue
                        else:
                            gt_0_count += 1
                    else:
                        gt = 1.0
                        if gt_1_count >= max_1_count:
                            continue
                        else:
                            gt_1_count += 1
                    tensor_gt = torch.tensor(gt, dtype=torch.float32)

                    obs1 = dataset["obs"][t0:t1]
                    obs2 = dataset["obs"][i:i+args.num_image]
                    concat_obs = [obs1, obs2]
                    tensor_obs = torch.tensor(np.array(concat_obs), dtype=torch.float32)
                    data ={"imgs":tensor_obs, "gt":tensor_gt}
                
                    with open(path, "wb") as f:
                        torch.save(data, f)

        print("end:" + bagfile_path)

if __name__ == "__main__":
    main()

