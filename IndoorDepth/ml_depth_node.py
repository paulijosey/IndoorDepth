# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    ml_depth_node.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Paul Joseph <paul.joseph@pbl.ee.ethz.ch    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/03 15:00:27 by Paul Joseph       #+#    #+#              #
#    Updated: 2024/04/04 08:47:41 by Paul Joseph      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from queue import Queue
import numpy as np
import torch
import cv2

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from gaze_msgs.msg import GazeStamped
from cv_bridge import CvBridge
 
# depth estimator imports
from IndoorDepth.inference_single_image import DepthEstimator

class IndoorDepth(Node):
    #      ____                _              _
    #     / ___|___  _ __  ___| |_ _   _  ___| |_ ___  _ __
    #    | |   / _ \| '_ \/ __| __| | | |/ __| __/ _ \| '__|
    #    | |__| (_) | | | \__ \ |_| |_| | (__| || (_) | |
    #     \____\___/|_| |_|___/\__|\__,_|\___|\__\___/|_|
    def __init__(self) -> None:
        super().__init__('ml_depth_node')
        # customizable params
        self.glassesNode = '/smart_glasses'    # node name for the smartglasses
        self.max_queue_size = 1     # max images saved in queue

        # init data queues
        self.glasses_gaze_queue = Queue(maxsize=self.max_queue_size)
        self.glasses_cam_queue = Queue(maxsize=self.max_queue_size)

        # cuda stuff (use GPU if available, else use CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

        # init depth estimator
        self.model_path = '/ws/src/IndoorDepth/IndoorDepth/models/IndoorDepth/' # hardcode for now
        self.depth_estimator = DepthEstimator(self.model_path, self.device)

        # init ROS stuff
        self.init_ros()
 
    #    ____      _ _ _                _        
    #   / ___|__ _| | | |__   __ _  ___| | _____ 
    #  | |   / _` | | | '_ \ / _` |/ __| |/ / __|
    #  | |__| (_| | | | |_) | (_| | (__|   <\__ \
    #   \____\__,_|_|_|_.__/ \__,_|\___|_|\_\___/
    def glasses_gaze_sub_callback(self, msg) -> None:
        self.queue_msg(msg, self.glasses_gaze_queue)

    def glasses_cam_sub_callback(self, msg) -> None:
        self.queue_msg(msg, self.glasses_cam_queue)
    
    def depth_glasses_pub_callback(self):
        self.estimate_depth()

    #    _____                 _   _                 
    #   |  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
    #   | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
    #   |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
    #   |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
    def estimate_depth(self):
        if (not self.glasses_gaze_queue.empty() and 
            not self.glasses_cam_queue.empty()):
            # get the latest image
            glasses_img, glasses_img_header = self.get_img_from_msg(self.glasses_cam_queue.get())
            glasses_gaze = self.get_gaze_from_msg(self.glasses_gaze_queue.get())
            # get the depth image
            glasses_depth_img, glasses_gaze_depth = self.depth_estimator.inference(glasses_img, glasses_gaze)
            # publish the depth image
            self.publish_depth_img(glasses_depth_img, glasses_img_header)

    #    ___       _ _   
    #   |_ _|_ __ (_) |_ 
    #    | || '_ \| | __|
    #    | || | | | | |_ 
    #   |___|_| |_|_|\__|

    def init_ros(self):
        #   use cv bridge to handle cv2 to ROS convertion
        self.cv_bridge = CvBridge()
        #   publisher for pretty pictures + gaze estimate
        self.depth_glasses_pub = self.create_publisher(Image, self.glassesNode + '/depth_img', 10)
        self.depth_glasses_pub_timer = self.create_timer(0.1, self.depth_glasses_pub_callback)

        #   subscriber for the gaze data
        self.glasses_gaze_sub = self.create_subscription(
            GazeStamped,
            self.glassesNode + '/gaze',
            self.glasses_gaze_sub_callback,
            10)
        self.glasses_gaze_sub # prevent unused variable warning
        #   subscriber for the cam data
        self.glasses_cam_sub = self.create_subscription(
            Image,
            self.glassesNode + '/cam_outward',
            self.glasses_cam_sub_callback,
            10)
        self.glasses_cam_sub # prevent unused variable warning

    #    _   _ _   _ _     
    #   | | | | |_(_) |___ 
    #   | | | | __| | / __|
    #   | |_| | |_| | \__ \
    #    \___/ \__|_|_|___/
    def queue_msg(self, msg, queue) -> None:
        '''
        Queue incoming ROS messages in queues for later use
        '''
        # check if we are full
        if(queue.qsize() >= self.max_queue_size):
            # pop oldest item
            queue.get()
        # add new item 
        queue.put(msg)

    def get_img_from_msg(self, msg: Image) -> np.array:
        return self.cv_bridge.imgmsg_to_cv2(msg), msg.header

    def get_gaze_from_msg(self, msg: GazeStamped) -> np.array:
        gaze = [msg.gaze.x, msg.gaze.y]
        return gaze
    
    def publish_depth_img(self, img: np.array, header) -> None:
        '''
        Publish the depth image to ROS
        '''
        msg = self.cv_bridge.cv2_to_imgmsg(img)
        msg.header = header
        self.depth_glasses_pub.publish(msg)

def main(args=None):
    '''
    Main! What more explanation do you need?
    '''
    # init for all ROS things
    rclpy.init(args=args)

    # init glasses (give an IP address if necessary! 
    #  check in neon companion android app)
    indoorDepth = IndoorDepth()

    # Spin ROS
    rclpy.spin(indoorDepth)

    # clean up
    rclpy.shutdown()

if __name__ == "__main__":
    main()
