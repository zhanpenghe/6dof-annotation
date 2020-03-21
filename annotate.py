import os.path as osp
import pickle
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import common3Dfunc as c3D
from utils import get_pointcloud


# Path for ground truth meshes
YCB_MESH_PATH = '../ycb/'

# Names for objects used for annotation
objs = [
    '010_potted_meat_can',
    '003_cracker_box',
    '006_mustard_bottle',
    '065-d_cups'
]

# Full path of the meshes
obj_gt_paths = [
    osp.join(YCB_MESH_PATH, obj, 'google_16k/textured.obj')
    for obj in objs
]


def load_gt_meshes(gt_paths):
    global all_pcd_list
    for path in gt_paths:
        all_pcd_list.append(o3d.io.read_triangle_mesh(path))

all_pcd_list = []
load_gt_meshes(obj_gt_paths)


# Object parameters
CURRENT_INDEX = 0
MAX_OBJ = len(obj_gt_paths)

# Annotation parameters
offset = 0.01
rot_step = 0.1 * np.pi
all_transformations = [c3D.makeTranslation4x4([0, 0, 0]) for _ in obj_gt_paths]

# pipline parameters
seq = 0
step = -1
path = './data/'
scene_mesh = None
visible_objects = True

view_size = 2.
view_bounds = np.array([view_size] * 3)


# Load scene data
def load_step_data(path, seq, step, real_data=False):
    if not real_data:
        depth_image = np.load('{}/{}_{}_depth.npy'.format(path, seq, step))[0]
        color_image = cv2.imread('{}/{}_{}_color_image.png'.format(path, seq, step))
        cam_intr = np.load('./data/ycb_data/camera_intr.npy')
        cam_pose = np.load('./data/ycb_data/camera_pose.npy')
    else:
        data_path = osp.join(path, 'data_{}_{}.pkl'.format(seq, step))
        # data_path = osp.join('./real_data/real_data/seq{}'.format(seq), 'data_{}.pkl'.format(step))
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            if 'depth_init' in data.keys():
                depth_scale = np.loadtxt('./real_data/real_data_new/camera_depth_scale.txt')
                cam_intr = np.load('./real_data/real_data_new/camera_intr.npy')
                cam_pose = np.loadtxt('./real_data/real_data_new/camera_pose.txt')
                depth_image = data['depth_init'] * depth_scale
                color_image = data['color_init']
                cam_pts, rgb_pts = get_pointcloud(color_image, depth_image, cam_intr)
                cam_pts = np.transpose(
                    np.dot(cam_pose[0:3, 0:3], np.transpose(cam_pts)) + np.tile(cam_pose[0:3, 3:], (1, cam_pts.shape[0])))
            else:
                print(data.keys())
                cam_intr_0 = np.load(osp.join(path, 'camera_intr.npy'))
                cam_intr_1 = np.load(osp.join(path, 'camera_intr_1.npy'))
                cam_pose_0 = np.loadtxt('./data/camera_pose.txt')
                cam_pose_1 = np.loadtxt('./data/camera_pose2.txt')
                depth_image_0 = data['depth_init_0'] * 9.919921874999999556e-01

                color_image_0 = data['color_init_0']
                cam_pts_0, rgb_pts_0 = get_pointcloud(color_image_0, depth_image_0, cam_intr_0)
                cam_pts_0 = np.transpose(
                    np.dot(cam_pose_0[0:3, 0:3], np.transpose(cam_pts_0)) + np.tile(cam_pose_0[0:3, 3:], (1, cam_pts_0.shape[0])))
                depth_image_1 = data['depth_init_1'] * 9.865234375000000444e-01
                color_image_1 = data['color_init_1']
                cam_pts_1, rgb_pts_1 = get_pointcloud(color_image_1, depth_image_1, cam_intr_1)
                cam_pts_1 = np.transpose(
                    np.dot(cam_pose_1[0:3, 0:3], np.transpose(cam_pts_1)) + np.tile(cam_pose_1[0:3, 3:], (1, cam_pts_1.shape[0])))

                return cam_pts_0, rgb_pts_0

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
        all_transformations[CURRENT_INDEX][0:3, 3] = translation[0:3,  3] + all_transformations[CURRENT_INDEX][0:3, 3]

    def translate_y_pos(vis):
        global CURRENT_INDEX, offset
        translation = c3D.makeTranslation4x4(np.asarray([0., offset, 0.]))
        pcd_list[CURRENT_INDEX].transform(translation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX][0:3, 3] = translation[0:3, 3] + all_transformations[CURRENT_INDEX][0:3, 3]
    
    def translate_z_pos(vis):
        global CURRENT_INDEX, offset
        translation = c3D.makeTranslation4x4(np.asarray([0., 0., offset]))
        pcd_list[CURRENT_INDEX].transform(translation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX][0:3, 3] = translation[0:3, 3] + all_transformations[CURRENT_INDEX][0:3, 3]

    def translate_z_neg(vis):
        global CURRENT_INDEX, offset
        translation = c3D.makeTranslation4x4(np.asarray([0., 0., -offset]))
        pcd_list[CURRENT_INDEX].transform(translation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX][0:3, 3] = translation[0:3, 3] + all_transformations[CURRENT_INDEX][0:3, 3]

    def rotate_1(vis):
        global CURRENT_INDEX, rot_step
        center = pcd_list[CURRENT_INDEX].get_center()
        rotation = c3D.ComputeTransformationMatrixAroundCentroid_MESH(center, rot_step, 0., 0.)
        pcd_list[CURRENT_INDEX].transform(rotation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX] = np.dot( rotation, all_transformations[CURRENT_INDEX] )

    def rotate_2(vis):
        global CURRENT_INDEX, rot_step
        # print('Rotation around roll axis')
        # rotation = R.from_euler('z', rot_step, degrees=False).as_matrix()
        center = pcd_list[CURRENT_INDEX].get_center()
        rotation = c3D.ComputeTransformationMatrixAroundCentroid_MESH(center, 0., rot_step, 0.)
        pcd_list[CURRENT_INDEX].transform(rotation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX] = np.dot(rotation, all_transformations[CURRENT_INDEX] )

    def rotate_3(vis):
        global CURRENT_INDEX, rot_step
        center = pcd_list[CURRENT_INDEX].get_center()
        rotation = c3D.ComputeTransformationMatrixAroundCentroid_MESH(center, 0., 0., rot_step)
        pcd_list[CURRENT_INDEX].transform(rotation)
        vis.update_geometry(pcd_list[CURRENT_INDEX])
        all_transformations[CURRENT_INDEX] = np.dot( rotation, all_transformations[CURRENT_INDEX] )

    def increase_idx(vis):
        global CURRENT_INDEX
        CURRENT_INDEX = (CURRENT_INDEX + 1) % MAX_OBJ

    def next_step(vis):
        global path, seq, step, scene_mesh, view_bounds
        step += 1
        if step > 0:
            vis.remove_geometry(scene_mesh)

        points = load_step_data(path, seq, step, real_data=True)
        if len(points) == 2:
            cam_pts, rgb_pts = points
            scene_mesh = o3d.geometry.PointCloud()

            rgb_pts = rgb_pts / 255.

            scene_mesh.points = o3d.utility.Vector3dVector(cam_pts)
            scene_mesh.colors = o3d.utility.Vector3dVector(rgb_pts)
            bounding_box = o3d.geometry.OrientedBoundingBox(
                np.array([0., 0., 0.]),
                np.identity(3),
                view_bounds)
            scene_mesh = scene_mesh.crop(bounding_box)
        else:
            cam_pts_0, cam_pts_1, rgb_pts_0, rgb_pts_1 = points
            rgb_pts_0 = rgb_pts_0 / 255.
            rgb_pts_1 = rgb_pts_1 / 255.

            cam_pts = np.vstack([cam_pts_0, cam_pts_1])
            rgb_pts = np.vstack([rgb_pts_0, rgb_pts_1])
            scene_mesh = o3d.geometry.PointCloud()
            
            scene_mesh.points = o3d.utility.Vector3dVector(cam_pts)
            scene_mesh.colors = o3d.utility.Vector3dVector(rgb_pts)

            # scene_mesh_1 = o3d.geometry.PointCloud()
            
            # scene_mesh_1.points = o3d.utility.Vector3dVector(cam_pts_1)
            # scene_mesh_1.colors = o3d.utility.Vector3dVector(rgb_pts_1)

            # scene_mesh = scene_mesh_0 + scene_mesh_1

        # vol_bnds = np.array([[0.244, 0.756],
        #             [-0.256, 0.256],
        #             [0.0, 0.192]])

        # scene_mesh.translate([-0.244, 0.256, 0.02])
        # scene_mesh.scale(250, center=False)

        # obs_path = osp.join('./real_data/real_data', '{}_{}_obs.ply'.format(seq, step))
        # o3d.io.write_point_cloud(obs_path, scene_mesh)
        vis.add_geometry(scene_mesh)

    def save_transform(vis):
        global path, seq, step, all_pcd_list, scene_mesh
        transform_path = osp.join(
            path, '{}_{}_tranform.pkl'.format(seq, step))
        with open(transform_path, 'wb') as f:
            pickle.dump(all_transformations, f)
            print('Transformation is saved to path:', transform_path)

        # # write meshes:
        # mesh_path = osp.join('./real_data/real_data', '{}_{}_annotation.ply'.format(seq, step))

        # # combine
        # v_len = 0
        # vertexes= []
        # triangles = []
        # colors = []
        # for obj in all_pcd_list:
        #     # vertexes.append(np.asarray(obj.vertices))
        #     # triangles.append(np.asarray(obj.triangles) + v_len)
        #     # colors.append(np.asarray(obj.texture))
        #     # v_len += vertexes[-1].shape[0]

        # vertexes = o3d.utility.Vector3dVector(np.vstack(vertexes))
        # triangles = o3d.utility.Vector3iVector(np.vstack(triangles))
        # colors = np.mean(colors, axis=0)
        # print(colors, np.asarray(colors).shape, np.vstack(vertexes).shape)

        # colors = o3d.geometry.Image(colors)

        # new_mesh = o3d.geometry.TriangleMesh(vertexes, triangles)
        # new_mesh.texture = o3d.geometry.Image(colors)
        # vis.add_geometry(new_mesh)
        # o3d.io.write_triangle_mesh(mesh_path, new_mesh)

        # # obs
        # obs_path = osp.join('./real_data/real_data', '{}_{}_obs.ply'.format(seq, step))
        # # o3d.io.write_triangle_mesh(obs_path, scene_mesh)

    def toggle_all_objects(vis):
        global all_pcd_list, visible_objects, MAX_OBJ
        for obj in all_pcd_list[0:MAX_OBJ]:
            if visible_objects:
                vis.remove_geometry(obj)
            else:
                vis.add_geometry(obj)
        visible_objects = not visible_objects

    key_to_callback = {}
    key_to_callback[ord("W")] = translate_y_pos
    key_to_callback[ord("S")] = translate_y_neg
    key_to_callback[ord("A")] = translate_x_neg
    key_to_callback[ord("D")] = translate_x_pos
    key_to_callback[ord("T")] = translate_z_pos
    key_to_callback[ord("G")] = translate_z_neg
    key_to_callback[ord("1")] = rotate_1
    key_to_callback[ord("2")] = rotate_2
    key_to_callback[ord("3")] = rotate_3
    key_to_callback[ord("N")] = increase_idx
    key_to_callback[ord("0")] = next_step
    key_to_callback[ord("9")] = save_transform
    key_to_callback[ord("V")] = toggle_all_objects

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
    print(
        'Runnning 6dof annotation {} with the following parameters:\n\t- sequence index: {}'.format(*sys.argv))
    seq = int(sys.argv[1])
    draw_geometry_with_key_callback(all_pcd_list)
