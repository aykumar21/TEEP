cmake_minimum_required(VERSION 3.5)
project(fault_injector_node)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(mavros_msgs REQUIRED)

# Install Python modules
install(PROGRAMS
  fault_injector_node/fault_injector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Export package
ament_package()
