### Quick Run Instructions

To quickly set up and run the self-driving project, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/fachrinnk4869/selfdriving
   cd selfdriving
   ```

2. **Clean Up Workspace**
   ```bash
   rm -rf logs build devel
   ```

3. **Build the Catkin Workspace**
   ```bash
   catkin build
   ```
4. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```
5. **Install model for yolov2 and polylannet**
   ```bash
   bash model.bash
   ```
6. **Run the Brio Camera Node**
   ```bash
   rosrun cv_camera cv_camera_node _cv_cap_prop_frame_width:=640 _cv_cap_prop_frame_height:=480
   ```

7. **Launch YOLOP Lane Detection**
   ```bash
   roslaunch yolop lane_camera.launch
   ```

### Installation and Setup

#### Install ROS Noetic

1. **Set up your sources.list**
   ```bash
   sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
   ```

2. **Set up your keys**
   ```bash
   sudo apt install curl # if you haven't already installed curl
   curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
   ```

3. **Update your package index**
   ```bash
   sudo apt update
   ```

4. **Install ROS Noetic full desktop version**
   ```bash
   sudo apt install ros-noetic-desktop-full
   ```

5. **Initialize rosdep**
   ```bash
   sudo rosdep init
   rosdep update
   ```

6. **Environment setup**
   ```bash
   echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

7. **Install dependencies for building packages**
   ```bash
   sudo apt install python3-rosinstall python3-rosinstall-generator python3-wstool build-essential ros-noetic-catkin python3-catkin-tools 
   ```

8. **Create and initialize a catkin workspace**
   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/
   catkin_make
   ```

#### Install `dependency`
1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```
2. **Install model for yolov2 and polylannet**
   ```bash
   bash model.bash
   ```

#### Install `ros-realsense`

1. **Install the `ros-realsense2-camera` package**
   ```bash
   sudo apt-get install ros-noetic-realsense2-camera
   ```

#### Install ZED SDK and `zed-ros-wrapper`

1. **Install the ZED SDK**
   Follow the installation guide from the [official ZED SDK documentation](https://www.stereolabs.com/docs/installation/linux/).

2. **Install the `zed-ros-wrapper`**
   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/src
   git clone --recursive https://github.com/stereolabs/zed-ros-wrapper.git
   cd ~/catkin_ws
   rosdep install --from-paths src --ignore-src -r -y
   catkin_make
   ```

3. **Source the workspace**
   ```bash
   echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

### Building Catkin Workspace

#### Build Catkin Workspace
```bash
catkin build
```

#### If Using Anaconda (Deactivate Anaconda)
Switch to native Python or specify the Python executable:
```bash
catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3
```

### Running Camera Nodes

#### RealSense Camera
```bash
roslaunch realsense2_camera rs_camera.launch
```

#### Brio Camera
Install and run `cv_camera`:
```bash
sudo apt install ros-noetic-cv-camera
rosrun cv_camera cv_camera_node _cv_cap_prop_frame_width:=640 _cv_cap_prop_frame_height:=480
```

#### ZED2i Camera
```bash
roslaunch zed_wrapper zed2i.launch
```

### Viewing Camera Images

To view the image from the camera topic:
```bash
rosrun image_view image_view image:=/cv_camera/image_raw
```

### Source Setup

Run source command to setup environment:
```bash
source devel/setup.bash
```

### Running Object Detection and Lane Detection

#### Object Detection with YOLO
```bash
roslaunch ultralytics_ros tracker.launch debug:=true
```

#### Lane Detection with YOLOPv2 without postprocessing (Video)
```bash
roslaunch yolop lane_raw.launch
```
#### Lane Detection with YOLOPv2 (Video + postprocessing)
```bash
roslaunch yolop lane_video.launch
```

#### Lane Detection with YOLOPv2 (Camera)
```bash
roslaunch yolop lane_camera.launch
```

### Output FPS to CSV

To print the FPS results to a CSV file:
```bash
roslaunch hasil hasil_fps.launch csv_filename_arg:=fps_yolop1.csv
```

### Running PolyLaneNet Order 1
```bash
roslaunch polylannet polylannet_video.launch order_arg:=1
```
### Running PolyLaneNet Order 2
```bash
roslaunch polylannet polylannet_video.launch order_arg:=2
```
### Running PolyLaneNet Order 3
```bash
roslaunch polylannet polylannet_video.launch order_arg:=3
```

This comprehensive guide will help you install ROS Noetic, `ros-realsense`, ZED, and run various camera nodes and detection algorithms using ROS.

### Lane Detection Code Integration

To ensure the correct camera topic is used in the lane detection code, replace the topic in the `yolopnode` package file `lane_detection_camera` at line 260 based on the camera being used.

```python
if __name__ == '__main__':
    rospy.init_node('lane_detection_camera')
    path = roslib.packages.get_pkg_dir("yolop")
    
    # Subscribe to the camera topic
    rospy.Subscriber("/cv_camera/image_raw", Image, image_callback)
    
    with torch.no_grad():
        # Initial call to detect in case of using image files
        if not rospy.has_param('/cv_camera/image_raw'):
            detect()
```