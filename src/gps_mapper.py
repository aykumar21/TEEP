import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import csv
import os
from camera_feed_inference import FloodClassifier  # uses the modular class

class GPSFloodMapper(Node):
    def __init__(self):
        super().__init__('gps_flood_mapper')
        self.bridge = CvBridge()
        self.classifier = FloodClassifier(model_path='flood_detection_model.pth')  # Load model

        self.declare_parameter('output_csv', 'flood_map.csv')
        self.output_csv = self.get_parameter('output_csv').get_parameter_value().string_value

        self.gps_data = None

        # Use BEST_EFFORT QoS for MAVROS GPS topic
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT

        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/mavros/global_position/raw/fix',
            self.gps_callback,
            qos_profile
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Prepare CSV file
        if not os.path.exists(self.output_csv):
            with open(self.output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Latitude', 'Longitude', 'Flood_Status'])

    def gps_callback(self, msg):
        self.gps_data = (msg.latitude, msg.longitude)

    def image_callback(self, msg):
        if self.gps_data is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {str(e)}")
            return

        label = self.classifier.predict(cv_image)

        # Save prediction and GPS to CSV
        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.gps_data[0], self.gps_data[1], label])
            self.get_logger().info(f"Logged: {self.gps_data} -> {label}")

        # Optional: Show window
        cv2.putText(cv_image, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Flood Detection", cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = GPSFloodMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
