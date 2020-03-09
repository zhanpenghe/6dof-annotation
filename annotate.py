import os.path as osp
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import common3Dfunc as c3D
from utils import get_pointcloud


# TODO have a flexible way for this
obj1 = o3d.io.read_triangle_mesh("../infogain-manipulation/ycb/003_cracker_box/google_16k/textured.obj")
obj2 = o3d.io.read_triangle_mesh("../infogain-manipulation/ycb/002_master_chef_can/google_16k/textured.obj")

obj_gt_paths = [
    '../infogain-manipulation/ycb/003_cracker_box/google_16k/textured.obj',
    '../infogain-manipulation/ycb/002_master_chef_can/google_16k/textured.obj',
]
all_pcd_list = [] 

# Object parameters
CURRENT_INDEX = 0
MAX_OBJ = len(obj_gt_paths)

# Annotation parameters
offset = 0.01
rot_step = 0.1 * np.pi
all_transformations = [c3D.makeTranslation4x4([0, 0, 0]) for _ in all_pcd_list]

# pipline parameters
seq = 0
step = -1
path = './data/ycb_data'
scene_mesh = None


def load_gt_meshes(gt_paths):
    global all_pcd_list
    for path in gt_paths:
        all_pcd_list.append(o3d.io.read_triangle_mesh(path))


# Load scene data
def load_step_data(path, seq, step):
    depth_image = np.load('{}/{}_{}_depth.npy'.format(path, seq, step))[0]
    color_image = cv2.imread('{}/{}_{}_color_image.png'.format(path, seq, step))
    cam_intr = np.load('./data/ycb_data/camera_intr.npy')
    cam_pose = np.load('./data/ycb_data/camera_pose.npy')

    cam_pts, rgb_pts = get_pointcloud(color_image, depth_image, cam_intr)

    cam_pts = np.transpose(
        np.dot(cam_pose[0:3, 0:3], np.transpose(cam_pts)) + np.tile(cam_pose[0:3, 3:], (1, cam_pts.shape[0])))

    return cam_pts, rgb_pts


def draw_geometry_with_key_callback(pcd_list):

    def scale_down(vis):
        global offset, rot_step
        offset /= 10.
        rot_step /= 10.

    def scale_up(vis):
        global offset, rot_step
        offset *= 10.
        rot_step *= 10.

    def translate_x_pos(vis):
        global CURRENT_INDEX, offset
        translation = c3D.makeTranslation4x4(np.asarray([offset, 0., 0.]))
        pcd_list[CURRENT_INDEX].transform(translation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX][0:3, 3] = translation[0:3, 3] + all_transformations[CURRENT_INDEX][0:3, 3]

    def translate_x_neg(vis):
        global CURRENT_INDEX, offset
        translation = c3D.makeTranslation4x4(np.asarray([-offset, 0., 0.]))
        pcd_list[CURRENT_INDEX].transform(translation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX][0:3, 3] = translation[0:3, 3] + all_transformations[CURRENT_INDEX][0:3, 3]

    def translate_y_neg(vis):
        global CURRENT_INDEX, offset
        translation = c3D.makeTranslation4x4(np.asarray([0., -offset, 0.]))
        pcd_list[CURRENT_INDEX].transform(translation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX][0:3, 3] = translation[0:3, 3] + all_transformations[CURRENT_INDEX][0:3, 3]

    def translate_y_pos(vis):
        global CURRENT_INDEX, offset
        translation = c3D.makeTranslation4x4(np.asarray([0., offset, 0.]))
        pcd_list[CURRENT_INDEX].transform(translation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        rotated_translation = np.dot(all_transformations[CURRENT_INDEX][0:3, 0:3], translation[0:3, 3])
        all_transformations[CURRENT_INDEX][0:3, 3] = rotated_translation + all_transformations[CURRENT_INDEX][0:3, 3]

    def rotate_1(vis):
        global CURRENT_INDEX, rot_step
        center = pcd_list[CURRENT_INDEX].get_center()
        rotation = c3D.ComputeTransformationMatrixAroundCentroid_MESH(center, rot_step, 0., 0.)
        pcd_list[CURRENT_INDEX].transform(rotation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX]= np.dot( rotation, all_transformations[CURRENT_INDEX] )

    def rotate_2(vis):
        global CURRENT_INDEX, rot_step
        # print('Rotation around roll axis')
        # rotation = R.from_euler('z', rot_step, degrees=False).as_matrix()
        center = pcd_list[CURRENT_INDEX].get_center()
        rotation = c3D.ComputeTransformationMatrixAroundCentroid_MESH(center, 0., rot_step, 0.)
        pcd_list[CURRENT_INDEX].transform(rotation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX]= np.dot( rotation, all_transformations[CURRENT_INDEX] )

    def rotate_3(vis):
        global CURRENT_INDEX, rot_step
        center = pcd_list[CURRENT_INDEX].get_center()
        rotation = c3D.ComputeTransformationMatrixAroundCentroid_MESH(center, 0., 0., rot_step)
        pcd_list[CURRENT_INDEX].transform(rotation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX]= np.dot( rotation, all_transformations[CURRENT_INDEX] )

    def increase_idx(vis):
        global CURRENT_INDEX
        CURRENT_INDEX = (CURRENT_INDEX + 1) % MAX_OBJ

    def next_step(vis):
        global path, seq, step, scene_mesh
        step += 1
        if step > 0:
            vis.remove_geometry(scene_mesh)

        cam_pts, rgb_pts = load_step_data(path, seq, step)
        scene_mesh = o3d.geometry.PointCloud()

        rgb_pts = rgb_pts / 255.

        scene_mesh.points = o3d.utility.Vector3dVector(cam_pts)
        scene_mesh.colors = o3d.utility.Vector3dVector(rgb_pts)
        vis.add_geometry(scene_mesh)

    def save_transform(vis):
        global path, seq, step
        transform_path = osp.join(path, '{}_{}_tranform.pkl')
        with open(transform_path, 'wb') as f:
            pickle.dump(all_transformations, f)

    key_to_callback = {}
    key_to_callback[ord("W")] = translate_y_pos
    key_to_callback[ord("S")] = translate_y_neg
    key_to_callback[ord("A")] = translate_x_neg
    key_to_callback[ord("D")] = translate_x_pos
    key_to_callback[ord("1")] = rotate_1
    key_to_callback[ord("2")] = rotate_2
    key_to_callback[ord("3")] = rotate_3
    key_to_callback[ord("N")] = increase_idx
    key_to_callback[ord("0")] = next_step
    key_to_callback[ord("9")] = save_transform

    key_to_callback[ord("+")] = scale_up
    key_to_callback[ord("-")] = scale_down
    o3d.visualization.draw_geometries_with_key_callbacks(pcd_list, key_to_callback)


def read_transform(transform_path):
    with open(transform_path, 'rb') as f:
        transform = pickle.load(f)
    print(transform)
    for i, t in enumerate(transform):
        all_pcd_list[i].transform(t)
    draw_geometry_with_key_callback(all_pcd_list)


if __name__ == "__main__":
    print(sys.argv)
    seq = int(sys.argv[1])
    load_gt_meshes(obj_gt_paths)
    draw_geometry_with_key_callback(all_pcd_list)
