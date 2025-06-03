import rclpy
from rclpy.node import Node
from geographic_msgs.msg import GeoPoseStamped
from std_msgs.msg import Header
from mavros_msgs.srv import SetMode
import time

WAYPOINTS = [
    {"lat": 47.3977500, "lon": 8.5456000, "alt": 533.85},
    {"lat": 47.3977800, "lon": 8.5454000, "alt": 535.00},
    {"lat": 47.3977900, "lon": 8.5456000, "alt": 537.00},
]

class WaypointPublisher(Node):
    def __init__(self):
        super().__init__('geo_waypoint_publisher')
        self.publisher_ = self.create_publisher(GeoPoseStamped, '/mavros/setpoint_position/global', 10)
        self.timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.waypoint_index = 0
        self.start_time = self.get_clock().now()
        self.hold_duration = 10  # seconds
        self.rtl_triggered = False

    def timer_callback(self):
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        if self.waypoint_index >= len(WAYPOINTS):
            if not self.rtl_triggered:
                self.trigger_rtl()
                self.rtl_triggered = True
            return

        if elapsed > self.hold_duration:
            self.waypoint_index += 1
            self.start_time = now
            return

        wp = WAYPOINTS[self.waypoint_index]
        msg = GeoPoseStamped()
        msg.header = Header()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.latitude = wp["lat"]
        msg.pose.position.longitude = wp["lon"]
        msg.pose.position.altitude = wp["alt"]
        msg.pose.orientation.w = 1.0  # No rotation

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing Waypoint {self.waypoint_index}: {wp}')

    def trigger_rtl(self):

        self.get_logger().info('All waypoints reached. Requesting RTL...')
        cli = self.create_client(SetMode, '/mavros/set_mode')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /mavros/set_mode service...')
        req = SetMode.Request()
        req.custom_mode = 'RTL'
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() and future.result().mode_sent:
            self.get_logger().info('RTL mode activated.')
        else:
            self.get_logger().warn('Failed to set RTL mode.')

def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
