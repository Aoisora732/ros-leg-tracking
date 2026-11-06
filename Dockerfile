FROM ros:noetic-ros-base-focal
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake libeigen3-dev libgtest-dev python3-numpy python3-matplotlib ros-noetic-rostest ros-noetic-tf2-geometry-msgs && rm -rf /var/lib/apt/lists/*
WORKDIR /ws/src/ros_leg_tracking
COPY . .
RUN /bin/bash -c 'source /opt/ros/noetic/setup.bash && cd /ws && catkin_make -DBUILD_ROS=ON -DCMAKE_BUILD_TYPE=Release'
CMD ["bash", "-c", "source /ws/devel/setup.bash && cd /ws/build/ros_leg_tracking && ctest --output-on-failure && rostest ros_leg_tracking integration.test"]
