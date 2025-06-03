import rclpy
from rclpy.node import Node
import subprocess
import threading
import time
import signal

class MAVROSWatchdog(Node):
    def __init__(self):
        super().__init__('mavros_watchdog')
        self.get_logger().info("🚀 Starting MAVROS watchdog with log monitoring...")

        self.mavros_process = None
        self.last_restart_time = 0
        self.restart_cooldown = 5  # seconds

        self.heartbeat_timeout = 3.0  # seconds
        self.last_heartbeat_time = time.time()

        self.launch_mavros()

    def launch_mavros(self):
        if self.mavros_process:
            self.get_logger().info("🛑 Killing old MAVROS process...")
            self.mavros_process.send_signal(signal.SIGINT)
            time.sleep(2)

        self.get_logger().info("🚁 Launching MAVROS...")
        self.mavros_process = subprocess.Popen(
            ['ros2', 'launch', 'mavros', 'px4.launch.py', 'fcu_url:=udp://:14540@127.0.0.1:14557'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Start a thread to monitor MAVROS logs
        threading.Thread(target=self.monitor_logs, daemon=True).start()

    def monitor_logs(self):
        for line in self.mavros_process.stdout:
            print(line.strip())  # Optional: also show in your terminal
            self.check_for_errors(line.strip())

            if "heartbeat timeout" in line.lower():
                self.last_heartbeat_time = time.time()  # Reset heartbeat timeout timer
                self.get_logger().warn("⚠️ Heartbeat timeout detected. Attempting recovery...")
                self.try_recovery()

    def check_for_errors(self, line):
        if "time jump detected" in line.lower():
            now = time.time()
            if now - self.last_restart_time > self.restart_cooldown:
                self.get_logger().warn("⏱ Time jump detected from MAVROS log. Restarting MAVROS...")
                self.last_restart_time = now
                self.launch_mavros()
            else:
                self.get_logger().info("⏱ Time jump detected but cooldown active.")

    def try_recovery(self):
        now = time.time()
        if now - self.last_heartbeat_time > self.heartbeat_timeout:
            self.get_logger().warn("❌ Heartbeat timeout persisted. Restarting MAVROS...")
            self.launch_mavros()
        else:
            self.get_logger().info("✅ Heartbeat timeout cleared.")

def main(args=None):
    rclpy.init(args=args)
    node = MAVROSWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node.mavros_process:
        node.mavros_process.terminate()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
