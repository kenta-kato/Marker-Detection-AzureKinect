from PointCloudManager import KinectManager


if __name__ == '__main__':
    num_id = [i for i in range(6)]
    AK = KinectManager(0.095, num_id)
    AK.run()
