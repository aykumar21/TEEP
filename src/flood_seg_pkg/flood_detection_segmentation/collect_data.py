#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class ImageCollector(Node):
    def __init__(self):
        super().__init__('image_collector_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Replace with your camera topic
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.image = None

        # Setup save directory
        dataset_dir = os.path.expanduser("~/TEEP/src/flood_seg_pkg/dataset")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(dataset_dir, f"session_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.img_count = 0

        self.get_logger().info(f"✓ Saving images to: {self.save_dir}")
        self.get_logger().info("Press 's' to save an image. Press 'q' to quit.")

    def listener_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("Drone Camera View (ROS)", self.image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and self.image is not None:
                filename = f"img_{self.img_count:04d}.jpg"
                path = os.path.join(self.save_dir, filename)
                cv2.imwrite(path, self.image)
                self.get_logger().info(f"[+] Saved {filename}")
                self.img_count += 1
            elif key == ord('q'):
                self.get_logger().info("Quitting...")
                rclpy.shutdown()
                cv2.destroyAllWindows()
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

