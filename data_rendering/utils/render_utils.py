import csv
import json
import os
import pickle
import random
import signal
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sapien.core as sapien
import transforms3d as t3d
from path import Path
from sapien.core import Pose
from data_rendering.utils.folder_paths import *

MAX_DEPTH = 2.0

RANDOM_SCALE = 0.5
SCALING_MIN = 0.5
SCALING_MAX = 1.5
METALLIC_MIN = 0.0
METALLIC_MAX = 0.8
ROUGHNESS_MIN = 0.0
ROUGHNESS_MAX = 0.8
SPECULAR_MIN = 0.0
SPECULAR_MAX = 0.8
TRANSMISSION_MIN = 0.0
TRANSMISSION_MAX = 1.0

PRIMITIVE_MIN = 10 # 25
PRIMITIVE_MAX = 150 # 100 # 50






with open(TEXTURE_LIST, "r") as f:
    texture_list = [line.strip() for line in f]
with open(TABLE_TEXTURE_LIST, "r") as f:
    table_texture_list = [line.strip() for line in f]
# with open(TEXTURE_SQ_LIST, "r") as f:
#     texture_sq_list = [line.strip() for line in f]
#
# with open(ENV_MAP_LIST, "r") as f:
#     env_list = [line.strip() for line in f]


def get_random_texture():
    random_file = random.choice(texture_list)
    path = os.path.join(TEXTURE_FOLDER, random_file)
    return path

def get_random_table_texture():
    random_file = random.choice(table_texture_list)
    path = os.path.join(TABLE_TEXTURE_FOLDER, random_file)
    print(path)
    return path

def get_random_sq_texture():
    random_file = random.choice(texture_sq_list)
    path = os.path.join(TEXTURE_SQ_FOLDER, random_file)
    return path


def get_random_env_file():
    random_file = random.choice(env_list)
    path = os.path.join(ENV_MAP_FOLDER, random_file)
    return path


def get_random_bin_texture():
    random_file = random.choice(texture_list)
    path = os.path.join(TEXTURE_FOLDER, random_file[:-4] + "_bin.png")
    return path


def load_kuafu_material(json_path, renderer: sapien.KuafuRenderer):
    with open(json_path, "r") as js:
        material_dict = json.load(js)

    js_dir = Path(json_path).dirname()
    object_material = renderer.create_material()
    object_material.set_base_color(material_dict["base_color"])
    if material_dict["diffuse_tex"]:
        object_material.set_diffuse_texture_from_file(str(js_dir / material_dict["diffuse_tex"]))
    object_material.set_emission(material_dict["emission"])
    object_material.set_ior(material_dict["ior"])
    object_material.set_metallic(material_dict["metallic"])
    if material_dict["metallic_tex"]:
        object_material.set_metallic_texture_from_file(str(js_dir / material_dict["metallic_tex"]))
    object_material.set_roughness(material_dict["roughness"])
    if material_dict["roughness_tex"]:
        object_material.set_roughness_texture_from_file(str(js_dir / material_dict["roughness_tex"]))
    object_material.set_specular(material_dict["specular"])
    object_material.set_transmission(material_dict["transmission"])
    if material_dict["transmission_tex"]:
        object_material.set_transmission_texture_from_file(str(js_dir / material_dict["transmission_tex"]))

    return object_material


def load_table(scene, table_dir, renderer, pose=Pose()):
    builder = scene.create_actor_builder()
    kuafu_material_path = os.path.join(table_dir, "kuafu_material.json")
    obj_material = load_kuafu_material(kuafu_material_path, renderer)
    builder.add_visual_from_file(os.path.join(table_dir, "optical_table.obj"), material=obj_material)
    table = builder.build_kinematic(name="table_kuafu")
    table.set_pose(pose)


def load_rand_table(scene, table_dir, renderer, pose=Pose()):
    builder = scene.create_actor_builder()
    kuafu_material_path = os.path.join(table_dir, "kuafu_material.json")
    with open(kuafu_material_path, "r") as js:
        material_dict = json.load(js)
    # Randomize material
    material = renderer.create_material()
    material.base_color = [np.random.rand(), np.random.rand(), np.random.rand(), 1.0]
    material.metallic = random.uniform(METALLIC_MIN, METALLIC_MAX)
    material.roughness = random.uniform(ROUGHNESS_MIN, ROUGHNESS_MAX)
    material.specular = random.uniform(SPECULAR_MIN, SPECULAR_MAX)
    material.ior = 1 + random.random()
    prob = random.random()
    """
    if prob < 0.2:
        material.set_transmission_texture_from_file(get_random_bin_texture()) # miniimagenet texture can be noisy
    elif prob < 0.7:
        material.set_diffuse_texture_from_file(get_random_texture())
    """
    if prob < 0.2:
        material.set_transmission_texture_from_file(get_random_bin_texture())
    elif prob < 0.7:
        material.set_diffuse_texture_from_file(get_random_table_texture())
    builder.add_visual_from_file(os.path.join(table_dir, "optical_table.obj"), material=material)
    table = builder.build_kinematic(name="table_kuafu")
    table.set_pose(pose)


def load_table_vk(scene, table_dir, pose=Pose()):
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(os.path.join(table_dir, "optical_table.obj"))
    table = builder.build_kinematic(name="table_vk")
    table.set_pose(pose)


def load_obj_vk(scene, obj_name, pose=Pose(), is_kinematic=False):
    builder = scene.create_actor_builder()
    builder.add_visual_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"))
    builder.add_multiple_collisions_from_file(os.path.join(OBJECT_DIR, obj_name, "collision_mesh.obj"))
    if is_kinematic:
        obj = builder.build_kinematic(name=obj_name)
    else:
        obj = builder.build(name=obj_name)
    obj.set_pose(pose)
    return obj


def create_realsense(camera_type: str, camera_name: str, default_resolution: tuple, camera_resolution: tuple,
                     camera_mount: sapien.ActorBase, scene: sapien.Scene, camera_k, camera_ir_k):
    scene.update_render()
    if camera_type == 'd415':
        # color fovy
        fov = 0.742501437664032
    elif camera_type == 'd435':
        fov = 0.78 # 0.742501437664032
    else:
        raise NotImplementedError()
    name = camera_name
    default_width, default_height = default_resolution
    width, height = camera_resolution
    camera_k[:2, :] = camera_k[:2, :] * np.array([[width / default_width], [height / default_height]])
    camera_ir_k[:2, :] = camera_ir_k[:2, :] * np.array([[width / default_width], [height / default_height]])

    tran_pose0 = sapien.Pose([0, 0, 0])
    if camera_type == 'd415':
        if "base" in camera_name:
            tran_pose1 = sapien.Pose(
                [-0.0008183810985, -0.0173809196, -0.002242552045],
                [9.99986449e-01, 5.69235052e-04, 1.23234267e-03, -5.02592655e-03],
            )
            tran_pose2 = sapien.Pose(
                [-0.0008183810985, -0.07214373, -0.002242552045],
                [9.99986449e-01, 5.69235052e-04, 1.23234267e-03, -5.02592655e-03],
            )
        else:
            tran_pose1 = sapien.Pose(
                [0.0002371878611, -0.0153303356, -0.002143536015],
                [0.9999952133080734, 0.0019029481504852343, -0.0003405963365571751, -0.0024158111293426307],
            )
            tran_pose2 = sapien.Pose(
                [0.0002371878611, -0.0702470843, -0.002143536015],
                [0.9999952133080734, 0.0019029481504852343, -0.0003405963365571751, -0.0024158111293426307],
            )
    elif camera_type == 'd435':
        tran_pose1 = sapien.Pose(
            [0.000156505, -0.0148977, -1.15315e-05], 
            [0.999964, -0.00674623, 0.00143998, -0.00484543]
        )
        tran_pose2 = sapien.Pose(
            [-0.000328569, -0.0650479, 0.000665889],
            [0.999964, -0.00674623, 0.00143998, -0.00484543]
        )
    else:
        raise NotImplementedError()

    camera0 = scene.add_mounted_camera(f"{name}", camera_mount, tran_pose0, width, height, 0, fov, 0.001, 100)
    camera0.set_perspective_parameters(
        0.01, 100.0, camera_k[0, 0], camera_k[1, 1], camera_k[0, 2], camera_k[1, 2], camera_k[0, 1]
    )

    camera1 = scene.add_mounted_camera(f"{name}_left", camera_mount, tran_pose1, width, height, 0, fov, 0.001, 100)
    camera1.set_perspective_parameters(
        0.01, 100.0, camera_ir_k[0, 0], camera_ir_k[1, 1], camera_ir_k[0, 2], camera_ir_k[1, 2], camera_ir_k[0, 1]
    )
    camera2 = scene.add_mounted_camera(f"{name}_right", camera_mount, tran_pose2, width, height, 0, fov, 0.001, 100)
    camera2.set_perspective_parameters(
        0.01, 100.0, camera_ir_k[0, 0], camera_ir_k[1, 1], camera_ir_k[0, 2], camera_ir_k[1, 2], camera_ir_k[0, 1]
    )

    return [camera0, camera1, camera2]


def spherical_pose(center: np.ndarray, radius: float, alpha: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # Obtain new position
    x_center = center[0]
    x_p = radius * np.sin(theta) * np.cos(alpha) + x_center
    y_center = center[1]
    y_p = radius * np.sin(theta) * np.sin(alpha) + y_center
    z_center = center[2]
    z_p = radius * np.cos(theta) + z_center
    cam_pos = np.array([x_p, y_p, z_p])

    # Obtain new rotation
    forward = (center - cam_pos) / np.linalg.norm(center - cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    return mat44


def cv2ex2pose(ex):
    ros2opencv = np.array(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
    )

    pose = np.linalg.inv(ex) @ ros2opencv

    from transforms3d.quaternions import mat2quat

    return sapien.Pose(pose[:3, 3], mat2quat(pose[:3, :3]))


def pose2cv2ex(pose):
    ros2opencv = np.array(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
    )
    ex = np.linalg.inv(np.matmul(pose, np.linalg.inv(ros2opencv)))
    return ex


def load_mesh_list(builder, mesh_list_file, renderer, material_name):
    mesh_dir = Path(mesh_list_file).dirname()
    fmesh = open(mesh_list_file, "r")
    for l in fmesh.readlines():
        mesh_name = l.strip()
        mesh_path = mesh_dir / mesh_name
        kuafu_material_path = mesh_dir / f"{mesh_name[:-4]}_{material_name}.json"
        if kuafu_material_path.exists():
            obj_material = load_kuafu_material(kuafu_material_path, renderer)
            builder.add_visual_from_file(mesh_path, material=obj_material)
        else:
            builder.add_visual_from_file(mesh_path)


def load_obj(scene, obj_name, renderer, pose=Pose(), is_kinematic=False, material_name="kuafu_material"):
    builder = scene.create_actor_builder()
    kuafu_material_path = os.path.join(OBJECT_DIR, obj_name, f"{material_name}.json")
    mesh_list_file = os.path.join(OBJECT_DIR, obj_name, "visual_mesh_list.txt")
    if os.path.exists(mesh_list_file):
        load_mesh_list(builder, mesh_list_file, renderer, material_name)
    elif os.path.exists(kuafu_material_path):
        obj_material = load_kuafu_material(kuafu_material_path, renderer)
        builder.add_visual_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"), material=obj_material)
    else:
        builder.add_visual_from_file(os.path.join(OBJECT_DIR, obj_name, "visual_mesh.obj"))
    builder.add_multiple_collisions_from_file(os.path.join(OBJECT_DIR, obj_name, "collision_mesh.obj"))
    if is_kinematic:
        obj = builder.build_kinematic(name=obj_name)
    else:
        obj = builder.build(name=obj_name)
    obj.set_pose(pose)

    return obj


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))
    theta, phi, z = randnums
    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z))
    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


# Get random pose on table, each grid on optical table is 0.12
def get_random_pose(h=0.02, clutter_option='regular'):
    # random_x = np.random.uniform(0.0, 0.6, 1)[0]
    # random_y = np.random.uniform(-0.3, 0.3, 1)[0]
    # random_z = np.random.uniform(0, 0.1, 1)[0]
    assert clutter_option in ['regular', 'smaller']
    if clutter_option == 'regular':
        random_x = np.random.uniform(-0.55, 1.4, 1)[0]
        random_y = np.random.uniform(-0.6, 1.0, 1)[0]
        random_z = np.random.uniform(0, 0.75, 1)[0]
    elif clutter_option == 'smaller':
        # smaller range
        random_x = np.random.uniform(-0.15, 0.7, 1)[0]
        random_y = np.random.uniform(-0.3, 0.5, 1)[0]
        random_z = np.random.uniform(0, 0.5, 1)[0]
    R = rand_rotation_matrix()
    T = np.hstack((R, np.array([[random_x], [random_y], [random_z]])))
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    pose = sapien.Pose.from_transformation_matrix(T)
    return pose


def sample_camera_pose_near_primitive(primitive_obj, center, min_radius=0.02, max_radius=0.25):
    r, l = primitive_obj['size']['r'], primitive_obj['size']['l']
    if primitive_obj['type'] == 'sphere':
        direction = np.random.uniform(-1, 1, 3)
        direction = direction / np.linalg.norm(direction + 1e-6)
        cam_pos = primitive_obj['pose'][:3, 3] + direction * (r + np.random.uniform(min_radius, max_radius))
    elif primitive_obj['type'] == 'capsule':
        # x-axis = half length direction
        if np.random.uniform() < 0.5:
            direction_circ = np.random.uniform(-1, 1, 2)
            direction_circ = direction_circ / np.linalg.norm(direction_circ + 1e-6)
            cam_pos_circ = direction_circ * (r + np.random.uniform(min_radius, max_radius))
            cam_pos = np.concatenate([np.random.uniform(-l, l, 1), cam_pos_circ])
        else:
            direction = np.random.uniform(-1, 1, 3)
            direction[0] = np.abs(direction[0])
            direction = direction / np.linalg.norm(direction + 1e-6)
            cam_pos = np.array([l, 0, 0]) + direction * (r + np.random.uniform(min_radius, max_radius))
            cam_pos = cam_pos * (np.random.uniform() < 0.5)
        capsule_to_cam = np.eye(4)
        capsule_to_cam[:3, 3] = cam_pos
        cam_pos = (primitive_obj['pose'] @ capsule_to_cam)[:3, 3]
    elif primitive_obj['type'] == 'box':
        direction = np.random.uniform(r + min_radius, r + max_radius, 3)
        direction = direction * np.random.choice([-1, 1], 3)
        box_to_cam = np.eye(4)
        box_to_cam[:3, 3] = direction
        cam_pos = (primitive_obj['pose'] @ box_to_cam)[:3, 3]
    
    if np.random.random() < 0.7:
        forward = primitive_obj['pose'][:3, 3] - cam_pos
    else:
        forward = center - cam_pos
    forward = forward / np.linalg.norm(forward)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    return mat44


def load_random_primitives_from_info(scene, renderer, idx, primitive_info):
    type = primitive_info["type"]

    builder = scene.create_actor_builder()

    # Randomize material
    primitive_material = primitive_info["material"]
    material = renderer.create_material()
    material.base_color = primitive_material["base_color"]
    material.metallic = primitive_material["metallic"]
    material.roughness = primitive_material["roughness"]
    material.specular = primitive_material["specular"]
    material.ior = primitive_material["ior"]

    r = primitive_info["size"]["r"]
    l = primitive_info["size"]["l"]
    pose = sapien.Pose.from_transformation_matrix(primitive_info["pose"])

    # Build
    if type == "sphere":
        builder.add_sphere_visual(radius=r, material=material)
        builder.add_sphere_collision(radius=r)
        s = builder.build_kinematic(name=str(idx))
        s.set_pose(pose)
    elif type == "capsule":
        builder.add_capsule_visual(radius=r, half_length=l, material=material)
        builder.add_capsule_collision(radius=r, half_length=l)
        s = builder.build_kinematic(name=str(idx))
        s.set_pose(pose)
    elif type == "box":
        builder.add_box_visual(half_size=[r, r, r], material=material)
        builder.add_box_collision(half_size=[r, r, r])
        s = builder.build_kinematic(name=str(idx))
        s.set_pose(pose)


def load_random_primitives(scene, renderer, idx):
    type = random.choice(["sphere", "capsule", "box"])

    builder = scene.create_actor_builder()

    # Randomize material
    material = renderer.create_material()
    material.base_color = [np.random.rand(), np.random.rand(), np.random.rand(), 1.0]
    material.metallic = random.uniform(METALLIC_MIN, METALLIC_MAX)
    material.roughness = random.uniform(ROUGHNESS_MIN, ROUGHNESS_MAX)
    material.specular = random.uniform(SPECULAR_MIN, SPECULAR_MAX)
    material.ior = 1 + random.random()
    prob = random.random()
    if prob < 0.1:
        material.transmission = TRANSMISSION_MAX
        # material.set_transmission_texture_from_file(get_random_texture())
    elif 0.1 <= prob < 0.6:
        material.set_diffuse_texture_from_file(get_random_texture())

    # Build
    if type == "sphere":
        r = 0.02 + np.random.rand() * 0.05
        l = 0
        builder.add_sphere_visual(radius=r, material=material)
        builder.add_sphere_collision(radius=r)
        s = builder.build_kinematic(name=str(idx))
        s.set_pose(get_random_pose())
    elif type == "capsule":
        r = 0.02 + np.random.rand() * 0.05
        l = 0.02 + np.random.rand() * 0.05
        builder.add_capsule_visual(radius=r, half_length=l, material=material)
        builder.add_capsule_collision(radius=r, half_length=l)
        s = builder.build_kinematic(name=str(idx))
        s.set_pose(get_random_pose())
    elif type == "box":
        r = 0.02 + np.random.rand() * 0.05
        l = 0
        builder.add_box_visual(half_size=[r, r, r], material=material)
        builder.add_box_collision(half_size=[r, r, r])
        s = builder.build_kinematic(name=str(idx))
        s.set_pose(get_random_pose())

    primitive_info = {
        f"obj_{idx}": {
            "idx": idx,
            "type": type,
            "material": {
                "base_color": material.base_color,
                "metallic": material.metallic,
                "roughness": material.roughness,
                "specular": material.specular,
                "ior": material.ior,
            },
            "size": {"r": r, "l": l},
            "pose": s.get_pose().to_transformation_matrix(),
        }
    }

    return primitive_info


def load_random_primitives_v2(scene, renderer, idx, clutter_option='regular'):
    type = random.choice(["sphere", "capsule", "box"])

    builder = scene.create_actor_builder()

    # Randomize material
    material = renderer.create_material()
    material.base_color = [np.random.rand(), np.random.rand(), np.random.rand(), 1.0]
    material.metallic = random.uniform(METALLIC_MIN, METALLIC_MAX)
    material.roughness = random.uniform(ROUGHNESS_MIN, ROUGHNESS_MAX)
    material.specular = random.uniform(SPECULAR_MIN, SPECULAR_MAX)
    material.ior = 1 + random.random()
    prob = random.random()
    if prob < 0.2:
        material.set_transmission_texture_from_file(get_random_bin_texture())
    elif prob < 0.7:
        material.set_diffuse_texture_from_file(get_random_texture())
    #
    # Build
    if type == "sphere":
        r = np.exp(np.random.uniform(np.log(0.004), np.log(0.10)))
        l = 0
        builder.add_sphere_visual(radius=r, material=material)
        builder.add_sphere_collision(radius=r)
        s = builder.build_kinematic(name=str(idx))
        s.set_pose(get_random_pose(clutter_option=clutter_option))
    elif type == "capsule":
        r = np.exp(np.random.uniform(np.log(0.004), np.log(0.10)))
        l = np.exp(np.random.uniform(np.log(0.004), np.log(0.20)))
        builder.add_capsule_visual(radius=r, half_length=l, material=material)
        builder.add_capsule_collision(radius=r, half_length=l)
        s = builder.build_kinematic(name=str(idx))
        s.set_pose(get_random_pose(clutter_option=clutter_option))
    elif type == "box":
        r = np.exp(np.random.uniform(np.log(0.004), np.log(0.10)))
        l = 0
        builder.add_box_visual(half_size=[r, r, r], material=material)
        builder.add_box_collision(half_size=[r, r, r])
        s = builder.build_kinematic(name=str(idx))
        s.set_pose(get_random_pose(clutter_option=clutter_option))

    primitive_info = {
        f"obj_{idx}": {
            "idx": idx,
            "type": type,
            "material": {
                "base_color": material.base_color,
                "metallic": material.metallic,
                "roughness": material.roughness,
                "specular": material.specular,
                "ior": material.ior,
            },
            "size": {"r": r, "l": l},
            "pose": s.get_pose().to_transformation_matrix(),
        }
    }

    return primitive_info

def check_camera_collision_with_primitive_dict(pos_arr, primitive_info, eps=0.01):
    # check whether the camera collides with any of the primitive shapes
    if not isinstance(pos_arr, list):
        pos_arr = [pos_arr]
    for pos in pos_arr:
        if pos[2] < 0.08:
            # avoid too low camera poses that cause penetration with table
            return True
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = pos
        for k, v in primitive_info.items():
            r, l = v['size']['r'], v['size']['l']
            this_eps = min(eps, max(r, l) / 2)
            if v['type'] == 'sphere':
                d = np.linalg.norm(pos - v['pose'][:3, 3])
                if d < r + this_eps:
                    # print("sphere", "rel_pos", pos - v['pose'][:3, 3], "r", r, "this_eps", this_eps, "d", d)
                    return True
            elif v['type'] == 'capsule':
                rel_pose = np.linalg.inv(v['pose']) @ cam_pose
                rel_pos = rel_pose[:3, 3]
                coll_cyl = np.linalg.norm(rel_pos[1:]) < r + this_eps
                coll_cyl = coll_cyl and (abs(rel_pos[0]) < l + this_eps)
                coll_sph = np.linalg.norm(rel_pos - np.array([l, 0, 0])) < r + this_eps
                coll_sph = coll_sph or (np.linalg.norm(rel_pos - np.array([-l, 0, 0])) < r + this_eps)
                if coll_cyl or coll_sph:
                    # if coll_cyl:
                        # print("coll_cyl", "rel_pos", rel_pos, "r", r, "l", l, "this_eps", this_eps, "np.linalg.norm(rel_pos[:2])", np.linalg.norm(rel_pos[:2]), "abs(rel_pos[-1])", abs(rel_pos[-1]))
                    # if coll_sph:
                        # print("coll_sph", "rel_pos", rel_pos, "r", r, "l", l, "this_eps", this_eps, "np.linalg.norm(rel_pos - np.array([0, 0, l]))", np.linalg.norm(rel_pos - np.array([0, 0, l])), "np.linalg.norm(rel_pos - np.array([0, 0, -l]))", np.linalg.norm(rel_pos - np.array([0, 0, -l])))
                    return True
            elif v['type'] == 'box':
                rel_pose = np.linalg.inv(v['pose']) @ cam_pose
                rel_pos = rel_pose[:3, 3]
                rel_pos_abs = np.abs(rel_pos)
                if np.all(rel_pos_abs < r + this_eps):
                    # print("box", "pos", pos, "rel_pos", rel_pos, "r", r, "this_eps", this_eps, v)
                    return True
    return False


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def visualize_depth(depth):
    cmap = plt.get_cmap("rainbow")
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    if len(depth.shape) == 3:
        depth = depth[..., 0]
    depth = np.clip(depth / MAX_DEPTH, 0.0, 1.0)
    vis_depth = cmap(depth)
    vis_depth = (vis_depth[:, :, :3] * 255.0).astype(np.uint8)
    vis_depth = cv2.cvtColor(vis_depth, cv2.COLOR_RGB2BGR)
    return vis_depth

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise Exception(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
