; Auto-generated. Do not edit!


(cl:in-package ultralytics_ros-msg)


;//! \htmlinclude YoloResult.msg.html

(cl:defclass <YoloResult> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (detections
    :reader detections
    :initarg :detections
    :type vision_msgs-msg:Detection2DArray
    :initform (cl:make-instance 'vision_msgs-msg:Detection2DArray))
   (masks
    :reader masks
    :initarg :masks
    :type (cl:vector sensor_msgs-msg:Image)
   :initform (cl:make-array 0 :element-type 'sensor_msgs-msg:Image :initial-element (cl:make-instance 'sensor_msgs-msg:Image))))
)

(cl:defclass YoloResult (<YoloResult>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <YoloResult>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'YoloResult)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name ultralytics_ros-msg:<YoloResult> is deprecated: use ultralytics_ros-msg:YoloResult instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <YoloResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ultralytics_ros-msg:header-val is deprecated.  Use ultralytics_ros-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'detections-val :lambda-list '(m))
(cl:defmethod detections-val ((m <YoloResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ultralytics_ros-msg:detections-val is deprecated.  Use ultralytics_ros-msg:detections instead.")
  (detections m))

(cl:ensure-generic-function 'masks-val :lambda-list '(m))
(cl:defmethod masks-val ((m <YoloResult>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader ultralytics_ros-msg:masks-val is deprecated.  Use ultralytics_ros-msg:masks instead.")
  (masks m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <YoloResult>) ostream)
  "Serializes a message object of type '<YoloResult>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'detections) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'masks))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'masks))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <YoloResult>) istream)
  "Deserializes a message object of type '<YoloResult>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'detections) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'masks) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'masks)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'sensor_msgs-msg:Image))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<YoloResult>)))
  "Returns string type for a message object of type '<YoloResult>"
  "ultralytics_ros/YoloResult")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'YoloResult)))
  "Returns string type for a message object of type 'YoloResult"
  "ultralytics_ros/YoloResult")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<YoloResult>)))
  "Returns md5sum for a message object of type '<YoloResult>"
  "1d3fa337e61f24dea5e6daf34d71b96f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'YoloResult)))
  "Returns md5sum for a message object of type 'YoloResult"
  "1d3fa337e61f24dea5e6daf34d71b96f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<YoloResult>)))
  "Returns full string definition for message of type '<YoloResult>"
  (cl:format cl:nil "std_msgs/Header header~%vision_msgs/Detection2DArray detections~%sensor_msgs/Image[] masks~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: vision_msgs/Detection2DArray~%# A list of 2D detections, for a multi-object 2D detector.~%~%Header header~%~%# A list of the detected proposals. A multi-proposal detector might generate~%#   this list with many candidate detections generated from a single input.~%Detection2D[] detections~%~%================================================================================~%MSG: vision_msgs/Detection2D~%# Defines a 2D detection result.~%#~%# This is similar to a 2D classification, but includes position information,~%#   allowing a classification result for a specific crop or image point to~%#   to be located in the larger image.~%~%Header header~%~%# Class probabilities~%ObjectHypothesisWithPose[] results~%~%# 2D bounding box surrounding the object.~%BoundingBox2D bbox~%~%# The 2D data that generated these results (i.e. region proposal cropped out of~%#   the image). Not required for all use cases, so it may be empty.~%sensor_msgs/Image source_img~%~%================================================================================~%MSG: vision_msgs/ObjectHypothesisWithPose~%# An object hypothesis that contains position information.~%~%# The unique numeric ID of object detected. To get additional information about~%#   this ID, such as its human-readable name, listeners should perform a lookup~%#   in a metadata database. See vision_msgs/VisionInfo.msg for more detail.~%int64 id~%~%# The probability or confidence value of the detected object. By convention,~%#   this value should lie in the range [0-1].~%float64 score~%~%# The 6D pose of the object hypothesis. This pose should be~%#   defined as the pose of some fixed reference point on the object, such a~%#   the geometric center of the bounding box or the center of mass of the~%#   object.~%# Note that this pose is not stamped; frame information can be defined by~%#   parent messages.~%# Also note that different classes predicted for the same input data may have~%#   different predicted 6D poses.~%geometry_msgs/PoseWithCovariance pose~%================================================================================~%MSG: geometry_msgs/PoseWithCovariance~%# This represents a pose in free space with uncertainty.~%~%Pose pose~%~%# Row-major representation of the 6x6 covariance matrix~%# The orientation parameters use a fixed-axis representation.~%# In order, the parameters are:~%# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)~%float64[36] covariance~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: vision_msgs/BoundingBox2D~%# A 2D bounding box that can be rotated about its center.~%# All dimensions are in pixels, but represented using floating-point~%#   values to allow sub-pixel precision. If an exact pixel crop is required~%#   for a rotated bounding box, it can be calculated using Bresenham's line~%#   algorithm.~%~%# The 2D position (in pixels) and orientation of the bounding box center.~%geometry_msgs/Pose2D center~%~%# The size (in pixels) of the bounding box surrounding the object relative~%#   to the pose of its center.~%float64 size_x~%float64 size_y~%~%================================================================================~%MSG: geometry_msgs/Pose2D~%# Deprecated~%# Please use the full 3D pose.~%~%# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.~%~%# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.~%~%~%# This expresses a position and orientation on a 2D manifold.~%~%float64 x~%float64 y~%float64 theta~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'YoloResult)))
  "Returns full string definition for message of type 'YoloResult"
  (cl:format cl:nil "std_msgs/Header header~%vision_msgs/Detection2DArray detections~%sensor_msgs/Image[] masks~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: vision_msgs/Detection2DArray~%# A list of 2D detections, for a multi-object 2D detector.~%~%Header header~%~%# A list of the detected proposals. A multi-proposal detector might generate~%#   this list with many candidate detections generated from a single input.~%Detection2D[] detections~%~%================================================================================~%MSG: vision_msgs/Detection2D~%# Defines a 2D detection result.~%#~%# This is similar to a 2D classification, but includes position information,~%#   allowing a classification result for a specific crop or image point to~%#   to be located in the larger image.~%~%Header header~%~%# Class probabilities~%ObjectHypothesisWithPose[] results~%~%# 2D bounding box surrounding the object.~%BoundingBox2D bbox~%~%# The 2D data that generated these results (i.e. region proposal cropped out of~%#   the image). Not required for all use cases, so it may be empty.~%sensor_msgs/Image source_img~%~%================================================================================~%MSG: vision_msgs/ObjectHypothesisWithPose~%# An object hypothesis that contains position information.~%~%# The unique numeric ID of object detected. To get additional information about~%#   this ID, such as its human-readable name, listeners should perform a lookup~%#   in a metadata database. See vision_msgs/VisionInfo.msg for more detail.~%int64 id~%~%# The probability or confidence value of the detected object. By convention,~%#   this value should lie in the range [0-1].~%float64 score~%~%# The 6D pose of the object hypothesis. This pose should be~%#   defined as the pose of some fixed reference point on the object, such a~%#   the geometric center of the bounding box or the center of mass of the~%#   object.~%# Note that this pose is not stamped; frame information can be defined by~%#   parent messages.~%# Also note that different classes predicted for the same input data may have~%#   different predicted 6D poses.~%geometry_msgs/PoseWithCovariance pose~%================================================================================~%MSG: geometry_msgs/PoseWithCovariance~%# This represents a pose in free space with uncertainty.~%~%Pose pose~%~%# Row-major representation of the 6x6 covariance matrix~%# The orientation parameters use a fixed-axis representation.~%# In order, the parameters are:~%# (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)~%float64[36] covariance~%~%================================================================================~%MSG: geometry_msgs/Pose~%# A representation of pose in free space, composed of position and orientation. ~%Point position~%Quaternion orientation~%~%================================================================================~%MSG: geometry_msgs/Point~%# This contains the position of a point in free space~%float64 x~%float64 y~%float64 z~%~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%================================================================================~%MSG: vision_msgs/BoundingBox2D~%# A 2D bounding box that can be rotated about its center.~%# All dimensions are in pixels, but represented using floating-point~%#   values to allow sub-pixel precision. If an exact pixel crop is required~%#   for a rotated bounding box, it can be calculated using Bresenham's line~%#   algorithm.~%~%# The 2D position (in pixels) and orientation of the bounding box center.~%geometry_msgs/Pose2D center~%~%# The size (in pixels) of the bounding box surrounding the object relative~%#   to the pose of its center.~%float64 size_x~%float64 size_y~%~%================================================================================~%MSG: geometry_msgs/Pose2D~%# Deprecated~%# Please use the full 3D pose.~%~%# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.~%~%# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.~%~%~%# This expresses a position and orientation on a 2D manifold.~%~%float64 x~%float64 y~%float64 theta~%~%================================================================================~%MSG: sensor_msgs/Image~%# This message contains an uncompressed image~%# (0, 0) is at top-left corner of image~%#~%~%Header header        # Header timestamp should be acquisition time of image~%                     # Header frame_id should be optical frame of camera~%                     # origin of frame should be optical center of camera~%                     # +x should point to the right in the image~%                     # +y should point down in the image~%                     # +z should point into to plane of the image~%                     # If the frame_id here and the frame_id of the CameraInfo~%                     # message associated with the image conflict~%                     # the behavior is undefined~%~%uint32 height         # image height, that is, number of rows~%uint32 width          # image width, that is, number of columns~%~%# The legal values for encoding are in file src/image_encodings.cpp~%# If you want to standardize a new string format, join~%# ros-users@lists.sourceforge.net and send an email proposing a new encoding.~%~%string encoding       # Encoding of pixels -- channel meaning, ordering, size~%                      # taken from the list of strings in include/sensor_msgs/image_encodings.h~%~%uint8 is_bigendian    # is this data bigendian?~%uint32 step           # Full row length in bytes~%uint8[] data          # actual matrix data, size is (step * rows)~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <YoloResult>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'detections))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'masks) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <YoloResult>))
  "Converts a ROS message object to a list"
  (cl:list 'YoloResult
    (cl:cons ':header (header msg))
    (cl:cons ':detections (detections msg))
    (cl:cons ':masks (masks msg))
))
