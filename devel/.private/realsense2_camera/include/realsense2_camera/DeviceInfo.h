// Generated by gencpp from file realsense2_camera/DeviceInfo.msg
// DO NOT EDIT!


#ifndef REALSENSE2_CAMERA_MESSAGE_DEVICEINFO_H
#define REALSENSE2_CAMERA_MESSAGE_DEVICEINFO_H

#include <ros/service_traits.h>


#include <realsense2_camera/DeviceInfoRequest.h>
#include <realsense2_camera/DeviceInfoResponse.h>


namespace realsense2_camera
{

struct DeviceInfo
{

typedef DeviceInfoRequest Request;
typedef DeviceInfoResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct DeviceInfo
} // namespace realsense2_camera


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::realsense2_camera::DeviceInfo > {
  static const char* value()
  {
    return "914e9cfa74a4f66f08c3fe1016943c1b";
  }

  static const char* value(const ::realsense2_camera::DeviceInfo&) { return value(); }
};

template<>
struct DataType< ::realsense2_camera::DeviceInfo > {
  static const char* value()
  {
    return "realsense2_camera/DeviceInfo";
  }

  static const char* value(const ::realsense2_camera::DeviceInfo&) { return value(); }
};


// service_traits::MD5Sum< ::realsense2_camera::DeviceInfoRequest> should match
// service_traits::MD5Sum< ::realsense2_camera::DeviceInfo >
template<>
struct MD5Sum< ::realsense2_camera::DeviceInfoRequest>
{
  static const char* value()
  {
    return MD5Sum< ::realsense2_camera::DeviceInfo >::value();
  }
  static const char* value(const ::realsense2_camera::DeviceInfoRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::realsense2_camera::DeviceInfoRequest> should match
// service_traits::DataType< ::realsense2_camera::DeviceInfo >
template<>
struct DataType< ::realsense2_camera::DeviceInfoRequest>
{
  static const char* value()
  {
    return DataType< ::realsense2_camera::DeviceInfo >::value();
  }
  static const char* value(const ::realsense2_camera::DeviceInfoRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::realsense2_camera::DeviceInfoResponse> should match
// service_traits::MD5Sum< ::realsense2_camera::DeviceInfo >
template<>
struct MD5Sum< ::realsense2_camera::DeviceInfoResponse>
{
  static const char* value()
  {
    return MD5Sum< ::realsense2_camera::DeviceInfo >::value();
  }
  static const char* value(const ::realsense2_camera::DeviceInfoResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::realsense2_camera::DeviceInfoResponse> should match
// service_traits::DataType< ::realsense2_camera::DeviceInfo >
template<>
struct DataType< ::realsense2_camera::DeviceInfoResponse>
{
  static const char* value()
  {
    return DataType< ::realsense2_camera::DeviceInfo >::value();
  }
  static const char* value(const ::realsense2_camera::DeviceInfoResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // REALSENSE2_CAMERA_MESSAGE_DEVICEINFO_H
