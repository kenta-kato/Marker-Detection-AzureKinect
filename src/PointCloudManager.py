import open3d as o3d
import pyk4a
from pyk4a import PyK4A, Config, CalibrationType, ColorResolution
import numpy as np
import cv2
import cv2.aruco as aruco


resolution_dict = {
    ColorResolution.RES_720P : [1280, 720],
    ColorResolution.RES_1080P: [1920, 1080],
    ColorResolution.RES_1440P: [2560, 1440],
    ColorResolution.RES_1536P: [2048, 1536],
    ColorResolution.RES_2160P: [3840, 2160],
    ColorResolution.RES_3072P: [4096, 3072]
}


class KinectManager:

    def __init__(self, marker_length, num_id):
        self.flag_exit = False

        # setup Azure Kinect
        self.k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                camera_fps=pyk4a.FPS.FPS_30,
                depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                synchronized_images_only=True,
            )
        )

        # setup point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.mesh.scale(0.2, center=self.mesh.get_center())
        self.prev_angle = np.array([0.0, 0.0, 0.0])

        # setup AR marker
        self.__setup_aruco(num_id)
        self.marker_length = marker_length

    def escape_callback(self):
        self.flag_exit = True
        return False

    def run(self):
        # setup geometries
        glfw_key_escape = 256
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.create_window('viewer', 1280, 720)
        print("Sensor initialized. Press [ESC] to exit.")
        vis_geometry_added = False
        reset_posture = True

        self.k4a.start()

        # Parameters required to generate a Point Cloud from an RGBD image.
        resolution = resolution_dict[self.k4a.calibration.color_resolution]
        kinect_intrinsics = self.k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
        distCoeffs = self.k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)

        pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(resolution[0],
                                                               resolution[1],
                                                               kinect_intrinsics[0][0],
                                                               kinect_intrinsics[1][1],
                                                               kinect_intrinsics[0][2],
                                                               kinect_intrinsics[1][2]
                                                               )

        while not self.flag_exit:
            rgbd = self.k4a.get_capture()
            if rgbd is None:
                continue

            color = np.asarray(rgbd.color).astype(np.uint8)
            depth = np.asarray(rgbd.transformed_depth).astype(np.float32) / 1000.0

            # The position and orientation of the AR marker are calculated from the acquired color image.
            color_aruco = cv2.cvtColor(color.copy(), cv2.COLOR_RGBA2RGB)
            xyz, angle, is_detect = self.__detect_marker(color_aruco, depth, self.marker_length, kinect_intrinsics, distCoeffs)

            # Generate a Point Cloud for visualization.
            color_pcd = cv2.cvtColor(color.copy(), cv2.COLOR_BGRA2RGB)
            img = o3d.geometry.Image(color_pcd)
            depth = o3d.geometry.Image(depth)

            cwd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth, depth_scale=1.0,
                                                                     depth_trunc=5.0, convert_rgb_to_intensity=False)
            tmp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(cwd, pinhole_intrinsics)

            self.pcd.points = tmp_pcd.points
            self.pcd.colors = tmp_pcd.colors

            # In this sample program, even when multiple markers are detected, only one marker is displayed on the geometry.
            xyz = xyz[0]
            angle = angle[0]

            # Display of a coordinate frame reflecting the position and orientation of the AR marker.
            self.mesh = self.mesh.translate(xyz, relative=False)

            # Rotate by the relative amount of change.
            if is_detect:
                tmp_change = np.deg2rad(angle - self.prev_angle)
                rotate_angle = self.mesh.get_rotation_matrix_from_xyz(tmp_change)
                self.mesh.rotate(rotate_angle, center=self.mesh.get_center())

                self.prev_angle = angle
                reset_posture = True
            else:
                # If no markers are detected, return to the initial state.
                if reset_posture:
                    tmp_change = np.deg2rad(-self.prev_angle)
                    rotate_angle = self.mesh.get_rotation_matrix_from_xyz(tmp_change)
                    self.mesh.rotate(rotate_angle, center=self.mesh.get_center())
                    reset_posture = False

            if vis_geometry_added:
                vis.update_geometry(self.pcd)
                vis.update_geometry(self.mesh)
            else:
                vis.add_geometry(self.pcd)
                vis.add_geometry(self.mesh)
                vis_geometry_added = True

            vis.poll_events()
            vis.update_renderer()

    def __setup_aruco(self, num_id):
        self.dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters_create()

        # The id list of markers to be detected.
        self.num_id = num_id

    def __detect_marker(self, img, depth, marker_length, intrinsics, distCoeffs):
        frame_markers = img.copy()

        corners, ids, _ = aruco.detectMarkers(img, self.dict_aruco, parameters=self.parameters)

        xyz = np.array([0.0, 0.0, 0.0])
        angle = np.array([0.0, 0.0, 0.0])
        is_detect = False

        position_list = []
        angle_list = []

        if not ids is None:
            for pos, detect_id in enumerate(ids):
                if detect_id in self.num_id:
                    is_detect = True
                    index = np.where(ids == self.num_id)[0][pos]
                    corner = corners[index][0]
                    center = np.mean(corner, axis=0)

                    # Convert the 2D center coordinates of the marker to the 3D coordinate system of Azure Kinect.
                    xyz = self.__convert_2d_to_3d(center[0], center[1], depth[int(center[1])][int(center[0])], intrinsics)

                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers([corner], marker_length, intrinsics, distCoeffs)
                    frame_markers = aruco.drawDetectedMarkers(frame_markers, corners, ids)
                    aruco.drawAxis(frame_markers, intrinsics, distCoeffs, rvec, tvec, 0.1)

                    # Convert rvec to euler
                    R = cv2.Rodrigues(rvec)[0]
                    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

                    if sy < 1e-6:
                        x = np.arctan2(-R[1, 2], R[1, 1])
                        y = np.arctan2(-R[2, 0], sy)
                        z = 0
                    else:
                        x = np.arctan2(R[2, 1], R[2, 2])
                        y = np.arctan2(-R[2, 0], sy)
                        z = np.arctan2(R[1, 0], R[0, 0])

                    pitch, yaw, roll = np.rad2deg([x, y, z])
                    angle = np.array([pitch, yaw, roll])

                    position_list.append(xyz)
                    angle_list.append(angle)
        else:
            position_list.append(xyz)
            angle_list.append(angle)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)
        thickness = 2

        for number, (tmp, pos) in enumerate(zip(angle_list, position_list)):
            show_text = 'id: {}, X: {:.2f}cm, Y: {:.2f}cm, Z: {:.2f}cm'.format(ids[number][0], pos[0] * 100.0,
                                                                               pos[1] * 100.0, pos[2] * 100.0)
            frame_markers = cv2.putText(frame_markers, show_text, (20, 30+100*number), font,
                                        font_scale, color, thickness, cv2.LINE_AA)
            show_text = 'id: {}, Pitch: {:.2f}, Yaw: {:.2f}, Roll: {:.2f}'.format(ids[number][0], tmp[0],
                                                                                  tmp[1], tmp[2])
            frame_markers = cv2.putText(frame_markers, show_text, (20, 80+100*number), font,
                                        font_scale, color, thickness, cv2.LINE_AA)

        cv2.imshow('frame', frame_markers)

        return position_list, angle_list, is_detect

    def __convert_2d_to_3d(self, u, v, z, K):
        K_inv = np.linalg.inv(K)
        camera_coordinate = np.array([u, v, 1])
        cc_ = camera_coordinate * z
        world_coordinate = np.dot(K_inv, cc_)

        return world_coordinate
