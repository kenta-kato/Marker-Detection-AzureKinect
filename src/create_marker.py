import cv2
import cv2.aruco as aruco

root_path = '../data/marker_4x4/'
dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)

for count in range(6):
    img_mark = aruco.drawMarker(dict_aruco, count, 200)
    cv2.imwrite(root_path + str(count) + '.png', img_mark)
