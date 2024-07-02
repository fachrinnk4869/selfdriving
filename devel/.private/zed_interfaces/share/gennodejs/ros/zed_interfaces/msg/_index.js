
"use strict";

let Skeleton3D = require('./Skeleton3D.js');
let ObjectsStamped = require('./ObjectsStamped.js');
let Keypoint3D = require('./Keypoint3D.js');
let Skeleton2D = require('./Skeleton2D.js');
let BoundingBox2Di = require('./BoundingBox2Di.js');
let Keypoint2Df = require('./Keypoint2Df.js');
let RGBDSensors = require('./RGBDSensors.js');
let PlaneStamped = require('./PlaneStamped.js');
let BoundingBox3D = require('./BoundingBox3D.js');
let Object = require('./Object.js');
let BoundingBox2Df = require('./BoundingBox2Df.js');
let PosTrackStatus = require('./PosTrackStatus.js');
let Keypoint2Di = require('./Keypoint2Di.js');

module.exports = {
  Skeleton3D: Skeleton3D,
  ObjectsStamped: ObjectsStamped,
  Keypoint3D: Keypoint3D,
  Skeleton2D: Skeleton2D,
  BoundingBox2Di: BoundingBox2Di,
  Keypoint2Df: Keypoint2Df,
  RGBDSensors: RGBDSensors,
  PlaneStamped: PlaneStamped,
  BoundingBox3D: BoundingBox3D,
  Object: Object,
  BoundingBox2Df: BoundingBox2Df,
  PosTrackStatus: PosTrackStatus,
  Keypoint2Di: Keypoint2Di,
};
