import rclpy
from rclpy.node import Node
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool
from rclpy.qos import QoSProfile
import subprocess

class FaultTolerantControlNode(Node):
    def __init__(self):
        super().__init__('ftc_node')
        self.qos = QoSProfile(depth=10)

        self.current_state = None
        self.last_heartbeat_time = self.get_clock().now()
        self.offboard_timeout = 3.0  # seconds

        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, self.qos)

        self.set_mode_cli = self.create_client(SetMode, '/mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, '/mavros/cmd/arming')

        while not self.set_mode_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /mavros/set_mode service...')
        while not self.arm_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /mavros/cmd/arming service...')

        self.create_timer(1.0, self.fault_monitor_callback)
        self.get_logger().info('✅ Fault Tolerant Control Node initialized.')

    def state_cb(self, msg):
        self.current_state = msg
        self.last_heartbeat_time = self.get_clock().now()

    def fault_monitor_callback(self):
        if self.current_state is None:
            return

        now = self.get_clock().now()
        elapsed = (now - self.last_heartbeat_time).nanoseconds / 1e9

        # 1. Check for OFFBOARD mode drop
        if self.current_state.mode != 'OFFBOARD':
            self.get_logger().warn('⚠️ OFFBOARD mode dropped. Attempting recovery...')
            if not self.try_recovery():
                self.get_logger().error('❌ OFFBOARD recovery failed. Executing fallback.')
                self.fallback_to_rtl()

        # 2. Check for MAVLink heartbeat timeout
        elif elapsed > self.offboard_timeout:
            self.get_logger().warn('⚠️ MAVLink heartbeat timeout. Attempting recovery...')
            if not self.try_recovery():
                self.get_logger().error('❌ Heartbeat recovery failed. Executing fallback.')
                self.fallback_to_rtl()

        # 3. Check for time jump errors from mavros
        if self.check_time_jump_log():
            self.get_logger().warn('⚠️ MAVROS Time Jump Detected! Executing fallback...')
            self.fallback_to_rtl()

    def try_recovery(self):
        req = SetMode.Request()
        req.custom_mode = 'OFFBOARD'

        future = self.set_mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().mode_sent:
            self.get_logger().info('✅ OFFBOARD recovery successful.')
            return True
        else:
            return False

    def fallback_to_rtl(self):
        req = SetMode.Request()
        req.custom_mode = 'AUTO.RTL'

        future = self.set_mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().mode_sent:
            self.get_logger().info('✅ Fallback executed: Switched to AUTO.RTL')
        else:
            self.get_logger().warn('⚠️ AUTO.RTL failed. Attempting AUTO.LAND instead.')
            self.fallback_to_land()

    def fallback_to_land(self):
        req = SetMode.Request()
        req.custom_mode = 'AUTO.LAND'

        future = self.set_mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().mode_sent:
            self.get_logger().info('✅ Switched to AUTO.LAND')
        else:
            self.get_logger().error('❌ Fallback to AUTO.LAND also failed!')

    def check_time_jump_log(self):
        # Scan latest journal logs for MAVROS time jump error
        try:
            result = subprocess.run(
                ["journalctl", "-n", "20", "--no-pager"],
                stdout=subprocess.PIPE,
                text=True
            )
            return 'Time jump detected' in result.stdout
        except Exception as e:
            self.get_logger().warn(f"⚠️ Failed to check journal logs: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = FaultTolerantControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
