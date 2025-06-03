#!/bin/bash

echo "Launching PX4 SITL with Gazebo Classic..."
cd ~/TEEP/src/PX4-Autopilot || exit
make px4_sitl gazebo-classic &

sleep 10

echo "Launching MAVROS with PX4 connection..."
gnome-terminal -- bash -c "ros2 launch mavros px4.launch fcu_url:='udp://:14540@127.0.0.1:14557'; exec bash"

sleep 5

echo "Starting FTC node..."
gnome-terminal -- bash -c "ros2 run ftc_node ftc_node; exec bash"

echo "Publishing OFFBOARD setpoint..."
gnome-terminal -- bash -c "ros2 topic pub -r 10 /mavros/setpoint_position/local geometry_msgs/msg/PoseStamped '{
  header: {frame_id: \"\"},
  pose: {
    position: {x: 6.0, y: 3.0, z: 4.0},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'; exec bash"

sleep 5

echo "Arming drone..."
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"

echo "Switching to OFFBOARD mode..."
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{base_mode: 0, custom_mode: 'OFFBOARD'}"

echo "Launching flood detection (Choose Mode):"
echo "1. ResNet-18 Classifier"
echo "2. UNet-18 Segmenter"
read -rp "Enter option [1/2]: " model_option

if [ "$model_option" == "1" ]; then
  echo "Running ResNet-18 Flood Classification..."
  gnome-terminal -- bash -c "cd ~/TEEP/src && python3 camera_feed_inference.py; exec bash"
elif [ "$model_option" == "2" ]; then
  echo "Running UNet-18 Flood Segmentation..."
  gnome-terminal -- bash -c "cd ~/TEEP/src/Flood_Detection_Segmentation && python3 flood_segmentation_inference.py; exec bash"
else
  echo "Invalid option. Skipping inference..."
fi

echo "Launching GPS Mapper..."
gnome-terminal -- bash -c "cd ~/TEEP/src && python3 gps_mapper.py; exec bash"

echo "✅ All systems running. Monitor topics with:"
echo "ros2 topic echo /mavros/local_position/pose"

