"""Modeling of a lidar sensor and a car.

The purpose is to perform an estimation of the amount of lidar point that could
be expected on a car. Knowing that we are using here a simplified model of car
as a 3D box.
"""
from typing import List
import numpy as np
import argparse
from sys import exit


class Car(object):
    """Model of a car."""

    def __init__(self, length: float, width: float, height: float) -> None:
        """Constructor of the car."""
        if (length <= 0) or (width <= 0) or (height <= 0):
            raise ValueError(
                f"Positive value expected. ({length}, {width}, {height})"
            )
        self.length: float = length
        self.width: float = width
        self.height: float = height
        self.center: np.array = np.array([0.0, 0.0, 0.0])
        self.corners: np.array = self._getBoxCoordinates()

    def _getRotationMatrix(self, angle: float, **kwargs) -> np.array:
        """Compute the rotation matrix from the yaw angle.

        For the simplified model of the car, we are only interested in rotating
        the car around the yaw axis. Thus we can consider the following
        rotation matrix:
                  _                            _
                 |                              |
                 | cos(angle)   -sin(angle)   0 |
            R =  | sin(angle)   cos(angle)    0 | 
                 |      0           0         1 |
                 |_                            _|
        
        Argument:
            angle: rotation angle in radian
        """
        return np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

    def _getTranslationVector(self, distance: float, **kwargs) -> np.array:
        """Compute the translation vector.
        
        Assuming that the x-axis is orthogonal to the car, this function just
        translate the car about the x-axis.
        """
        return np.array([distance, 0.0, 0.0])

    @property
    def distance(self) -> float:
        """Return the distance from the sensor."""
        # The x-axis coordinate of the center of the box is the distance to
        # the sensor. Assumming that the x-axis pass through the center of the
        # box and is orthogonal to it.
        return self.center[0]

    def move(self, distance: float, **kwargs) -> None:
        """Translate the car to a distance from the sensor.
        
        Argument:
            distance: how far the car should move from the sensor (in meter)
        """
        # Translation vector
        T = self._getTranslationVector(distance)
        self.center = self.center + T
        self.corners = self.corners + T

    def rotate(self, angle: float, **kwargs) -> None:
        """Rotate the car for an angle.
        
        Argument:
            angle: rotation angle in degree
        """
        # convert angle into radian
        angle = angle * np.pi / 180
        # Rotation matrix
        R = self._getRotationMatrix(angle)
        corners_t = np.dot(R, np.transpose(self.corners))
        self.corners = np.transpose(corners_t)
        self.center = np.dot(R, self.center)

    def _getBoxCoordinates(self) -> np.array:
        """Return the coordinates of the corners of the 3D-box.
        
        The 3D-bix is the model of the car.
        
                      z
                      |  /x
                      | /
                y_____|/

                c1__________ c2               
                /|         /|
             c4/_|________/ |     
               | /--------|-/ c6
             c8|/_________|/ c7
                <--------->
                  w: width

            cx: corner order number

        Method to compute corners coordinates:
            corner 1 (c1):
                x: CenterX + length/2
                y: CenterY + width/2
                z: CenterZ + height/2
            corner 2 (c2):
                x: CenterX + length/2
                y: CenterY - width/2
                z: CenterZ + height/2
            corner 3 (c3):
                x: CenterX - length/2
                y: CenterY - width/2
                z: CenterZ + height/2
            corner 4 (c4):
                x: CenterX - length/2
                y: CenterY + width/2
                z: CenterZ + height/2

            corner 5 (c5):
                x: CenterX + length/2
                y: CenterY + width/2
                z: CenterZ - height/2
            corner 6 (c6):
                x: CenterX + length/2
                y: CenterY - width/2
                z: CenterZ - height/2
            corner 7 (c7):
                x: CenterX - length/2
                y: CenterY - width/2
                z: CenterZ - height/2
            corner 8 (c8):
                x: CenterX - length/2
                y: CenterY + width/2
                z: CenterZ - height/2

        Return: array of corners coordinates
                c1.x, c1.y, c1.z
                c2.x, c2.y, c2.z
                c3.x, c3.y, c3.z
        """
        corners_coordinates = np.array([
            [self.length/2, self.width/2, self.height/2],       # c1
            [self.length/2, -self.width/2, self.height/2],      # c2
            [-self.length/2, -self.width/2, self.height/2],     # c3
            [-self.length/2, self.width/2, self.height/2],      # c4
            [self.length/2, self.width/2, -self.height/2],      # c5
            [self.length/2, -self.width/2, -self.height/2],     # c6
            [-self.length/2, -self.width/2, -self.height/2],    # c7
            [-self.length/2, self.width/2, -self.height/2],     # c8
        ])
        return corners_coordinates + self.center

    @property
    def exposure(self) -> np.array:
        """Return the dimensions of the exposed view of the car.
        
        The expose view of the car has a rectangular form.

        Return: [width, height]
        """
        return np.array([
            abs(max(self.corners[:, 1]) - min(self.corners[:, 1])),
            self.height,
        ])


class LidarSensor(object):
    """Model of the lidar sensor.

    Attributes:
        range: Range of the sensor
        vres: vertical resolution of the sensor in degree
        hres: horizontal resolution of the sensor in degree
    """
    VERTICAL_VIEW: float = 15.0       # degree
    HORIZONTAL_VIEW: float = 360.0    # degree

    def __init__(self, srange: float, vres: float, hres: float,
                 **kwargs) -> None:
        """Constructor of the Lidar sensor.
        
        Parameters:
            srange: sensor range
            vres: vertical resolution
            hres: horizontal resolution
        """
        if (srange <= 0) or (vres <= 0) or (hres <= 0):
            raise ValueError(
                f"Lidar parameters must have positive values."
                " range: {srange}, vres: {vres}, hres: {hres}"
            )
        self.range = srange
        self.vres = vres
        self.hres = hres
        self.vertical_view = kwargs.get("vertical_view", self.VERTICAL_VIEW)
        self.horizontal_view = kwargs.get(
            "horizontal_view",
            self.HORIZONTAL_VIEW,
        )

    def estimatePointCloud(self, car: Car) -> int:
        """Return the number lidar points expected on the car."""
        if car.distance > self.range:
            # If the car is over the range of the sensor
            # No lidar point must be expected
            return 0
        CONVERSION_RATION: float = 180.0 / np.pi
        width, height = car.exposure
        z_angle = min(
            2.0 * np.arctan2(height, 2.0 * car.distance) * CONVERSION_RATION,
            self.vertical_view,
        )
        y_angle = min(
            2.0 * np.arctan2(width, 2.0 * car.distance) * CONVERSION_RATION,
            self.horizontal_view,
        )
        return int(np.floor(z_angle/self.vres) * np.floor(y_angle/self.hres))


if __name__ == "__main__":
    CAR_WIDTH: float = 1.762    # meter
    CAR_HEIGHT: float = 1.59    # meter
    CAR_LENGTH: float = 4.391   # meter

    LIDAR_RANGE: float = 100    # meters
    LIDAR_VRES: float = 2.0     # degree
    LIDAR_HRES: float = 0.2     # degree

    # CLI parser
    parser = argparse.ArgumentParser(
        prog="lisim",
        description="Lidar simulator on 3D model of a car",
    )
    parser.add_argument(
        "--width",
        type=float,
        help="Width of the car",
        default=CAR_WIDTH,
    )
    parser.add_argument(
        "--length",
        type=float,
        help="Length of the car",
        default=CAR_LENGTH,
    )
    parser.add_argument(
        "--height",
        type=float,
        help="Height of the car",
        default=CAR_HEIGHT,
    )
    parser.add_argument(
        "-r", "--range",
        type=float,
        help="Range of the lidar sensor",
        default=LIDAR_RANGE,
    )
    parser.add_argument(
        "-hres", "--horizontal-resolution",
        type=float,
        help="Horizontal resolution of the lidar sensor",
        default=LIDAR_HRES,
    )
    parser.add_argument(
        "-vres", "--vertical-resolution",
        type=float,
        help="Vertical resolution of the lidar sensor",
        default=LIDAR_VRES,
    )

    args = parser.parse_args()

    for arg in [attr for attr in dir(args) if not ("_" in attr)]:
        if getattr(args, arg) < 0:
            print(
                f"Arguments must be positive. Get {arg}: {getattr(args, arg)}"
            )
            exit(1)

    distances: List[float] = [5.0, 10.0, 15.0, 20.0]    # meters
    angles: List[float] = [0.0, 45.0, 90.0]             # degrees

    lidar = LidarSensor(
        args.range,
        args.vertical_resolution,
        args.horizontal_resolution,
    )

    print(f"Car (length: {args.length} m; width: {args.width} m; ", end="")
    print(f"height: {args.height} m)", end="\n\n")

    print(" ------------------------------------------------------ ")
    print("|    Distance(m)  |   Angle(deg)   |    Lidar points   |")
    for distance in distances:
        for angle in angles:
            car: Car = Car(args.length, args.width, args.height)
            car.rotate(angle)
            car.move(distance)
            print("|-----------------|----------------|-------------------|")
            print(f"|     {distance:5.1f}       |", end="")
            print(f"     {angle:5.1f}      |     ", end="")
            print(f"{lidar.estimatePointCloud(car):6d}        |")
    print(" ------------------------------------------------------ ")
