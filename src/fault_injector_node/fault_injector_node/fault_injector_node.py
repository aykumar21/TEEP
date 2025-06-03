# File: fault_injector_node.py

import rclpy
from rclpy.node import Node
from mavros_msgs.srv import SetMode
import time

class FaultInjector(Node):
    def __init__(self):
        super().__init__('fault_injector_node')

        # Create service client to set mode
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        while not self.set_mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /mavros/set_mode service...')

        # Parameters (you can change these!)
        self.heartbeat_timeout_time = 20.0  # seconds after start to simulate heartbeat timeout
        self.offboard_drop_time = 30.0       # seconds after start to simulate OFFBOARD drop
        self.start_time = self.get_clock().now().seconds_nanoseconds()[0]

        # Timer to check time and inject faults
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.heartbeat_timeout_done = False
        self.offboard_drop_done = False

    def timer_callback(self):
        current_time = self.get_clock().now().seconds_nanoseconds()[0]
        elapsed_time = current_time - self.start_time

        # Simulate MAVLink heartbeat timeout
        if not self.heartbeat_timeout_done and elapsed_time >= self.heartbeat_timeout_time:
            self.get_logger().info('Simulating MAVLink heartbeat timeout... (You should stop heartbeat manually or ignore this)')
            # In real SITL, you can simulate by stopping MAVROS heartbeat publisher or killing connection
            self.heartbeat_timeout_done = True

        # Simulate OFFBOARD mode drop
        if not self.offboard_drop_done and elapsed_time >= self.offboard_drop_time:
            self.get_logger().info('Switching to MANUAL mode to simulate OFFBOARD drop...')
            self.set_mode('MANUAL')
            self.offboard_drop_done = True

    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Mode change response: {future.result().mode_sent}")
        else:
            self.get_logger().error('Failed to call set_mode service')

def main(args=None):
    rclpy.init(args=args)
    node = FaultInjector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
