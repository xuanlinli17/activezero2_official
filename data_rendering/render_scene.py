import os
import os.path as osp
import sys

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)
from loguru import logger
from PIL import Image

from data_rendering.utils.render_utils import *


def render_scene(
    sim: sapien.Engine,
    renderer: sapien.KuafuRenderer,
    scene_id,
    repo_root,
    target_root,
    camera_type,
    camera_resolution,
    spp,
    num_views,
    rand_pattern,
    fixed_angle,
    primitives,
    primitives_v2,
    rand_lighting,
    rand_table,
    rand_env,
):
    materials_root = os.path.join(repo_root, "data_rendering/materials")

    scene_config = sapien.SceneConfig()
    scene_config.solver_iterations = 25
    scene_config.solver_velocity_iterations = 2
    scene_config.enable_pcm = False
    scene_config.default_restitution = 0
    scene_config.default_dynamic_friction = 0.5
    scene_config.default_static_friction = 0.5
    scene = sim.create_scene(scene_config)
    scene.set_timestep(1 / 240)

    if not rand_env:
        ground_material = renderer.create_material()
        ground_material.base_color = np.array([10, 10, 10, 256]) / 256
        ground_material.specular = 0.5
        scene.add_ground(-2.0, render_material=ground_material)

    table_pose_np = np.loadtxt(os.path.join(repo_root, "data_rendering/materials/optical_table/pose.txt"))
    table_pose = sapien.Pose(table_pose_np[:3], table_pose_np[3:])

    if rand_table:
        load_rand_table(scene, os.path.join(repo_root, "data_rendering/materials/optical_table"), renderer, table_pose)
    else:
        load_table(scene, os.path.join(repo_root, "data_rendering/materials/optical_table"), renderer, table_pose)

    # Add camera
    if camera_type == 'd415':
        default_resolution = (1920, 1080)
        cam_intrinsic_base = np.loadtxt(os.path.join(materials_root, "cam_intrinsic_base.txt")) # intrinsic under default resolution
        cam_ir_intrinsic_base = np.loadtxt(os.path.join(materials_root, "cam_ir_intrinsic_base.txt"))
        cam_intrinsic_hand = np.loadtxt(os.path.join(materials_root, "cam_intrinsic_hand.txt"))
        cam_ir_intrinsic_hand = np.loadtxt(os.path.join(materials_root, "cam_ir_intrinsic_hand.txt"))
        cam_irL_rel_extrinsic_base = np.loadtxt(
            os.path.join(materials_root, "cam_irL_rel_extrinsic_base.txt")
        )  # camL -> cam0
        cam_irR_rel_extrinsic_base = np.loadtxt(
            os.path.join(materials_root, "cam_irR_rel_extrinsic_base.txt")
        )  # camR -> cam0
        cam_irL_rel_extrinsic_hand = np.loadtxt(
            os.path.join(materials_root, "cam_irL_rel_extrinsic_hand.txt")
        )  # camL -> cam0
        cam_irR_rel_extrinsic_hand = np.loadtxt(
            os.path.join(materials_root, "cam_irR_rel_extrinsic_hand.txt")
        )  # camR -> cam0
    elif camera_type == 'd435':
        default_resolution = (848, 480)
        cam_intrinsic_base = np.array([[605.12158203125,     0., 424.5927734375], [0.,    604.905517578125, 236.668975830078], [0, 0, 1]]) # intrinsic under default resolution
        cam_ir_intrinsic_base = np.array([[430.139801025391,   0., 425.162841796875], [0.,   430.139801025391, 235.276519775391], [0, 0, 1]])
        cam_intrinsic_hand = np.array(cam_intrinsic_base)
        cam_ir_intrinsic_hand = np.array(cam_ir_intrinsic_base)
        cam_irL_rel_extrinsic_base = np.array([[0.9999489188194275, 0.009671091102063656, 0.0029452370945364237, 0.00015650546993128955],
                                              [-0.009709948673844337, 0.999862015247345, 0.013478035107254982, -0.014897654764354229],
                                              [-0.0028144833631813526, -0.013505944050848484, 0.9999048113822937, -1.1531494237715378e-05],
                                              [0.0, 0.0, 0.0, 1.0]])
        cam_irR_rel_extrinsic_base = np.array([[0.9999489188194275, 0.009671091102063656, 0.0029452370945364237, -0.0003285693528596312],
                                              [-0.009709948673844337, 0.999862015247345, 0.013478035107254982, -0.06504792720079422],
                                              [-0.0028144833631813526, -0.013505944050848484, 0.9999048113822937, 0.0006658887723460793],
                                              [0.0, 0.0, 0.0, 1.0]])
        cam_irL_rel_extrinsic_hand = np.array(cam_irL_rel_extrinsic_base)
        cam_irR_rel_extrinsic_hand = np.array(cam_irR_rel_extrinsic_base)
    else:
        raise NotImplementedError()

    builder = scene.create_actor_builder()
    cam_mount = builder.build_kinematic(name="real_camera")
    if fixed_angle:
        # reproduce IJRR
        base_cam_rgb, base_cam_irl, base_cam_irr = create_realsense(
            camera_type, "real_camera_base", default_resolution, camera_resolution,
            cam_mount, scene, cam_intrinsic_base, cam_ir_intrinsic_base
        )

    hand_cam_rgb, hand_cam_irl, hand_cam_irr = create_realsense(
        camera_type, "real_camera_hand", default_resolution, camera_resolution,
        cam_mount, scene, cam_intrinsic_hand, cam_ir_intrinsic_hand
    )

    # Add lights
    """
    if rand_env:
        ambient_light = np.random.rand(3)
        scene.set_ambient_light(ambient_light)
        scene.set_environment_map(get_random_env_file())

        # change light
        def lights_on():
            ambient_light = np.random.rand(2)
            scene.set_ambient_light(ambient_light)
            scene.set_environment_map(get_random_env_file())

            alight.set_color([0.0, 0.0, 0.0])

        def lights_off():
            ambient_light = np.random.rand(3) * 0.05
            scene.set_ambient_light(ambient_light)
            alight_color = np.random.rand(3) * np.array((60, 20, 20)) + np.array([30, 10, 10])
            scene.set_environment_map(get_random_env_file())

            alight.set_color(alight_color)

        def light_off_without_alight():
            alight.set_color([0.0, 0.0, 0.0])
    """
    if rand_env:
        scene.set_environment_map(get_random_env_file())

    if rand_lighting:
        ambient_light = np.random.rand(3)
        scene.set_ambient_light(ambient_light)
        height = np.random.rand() + 2
        light_1_color = np.random.rand(3) * 20 + 20
        light_2_color = np.random.rand(3) * 10 + 5
        light_3_color = np.random.rand(3) * 10 + 5
        light_4_color = np.random.rand(3) * 10 + 5
        light_5_color = np.random.rand(3) * 10 + 5
        plight1 = scene.add_point_light([-0.3, -0.3, height], light_1_color)
        plight2 = scene.add_point_light([2, -2, height], light_2_color)
        plight3 = scene.add_point_light([-2, 2, height], light_3_color)
        plight4 = scene.add_point_light([2, 2, height], light_4_color)
        plight5 = scene.add_point_light([-2, -2, height], light_5_color)

        # change light
        def lights_on():
            ambient_light = np.random.rand(3)
            scene.set_ambient_light(ambient_light)
            light_1_color = np.random.rand(3) * 20 + 20
            light_2_color = np.random.rand(3) * 10 + 5
            light_3_color = np.random.rand(3) * 10 + 5
            light_4_color = np.random.rand(3) * 10 + 5
            light_5_color = np.random.rand(3) * 10 + 5
            plight1.set_color(light_1_color)
            plight2.set_color(light_2_color)
            plight3.set_color(light_3_color)
            plight4.set_color(light_4_color)
            plight5.set_color(light_5_color)
            alight.set_color([0.0, 0.0, 0.0])

        def lights_off():
            ambient_light = np.random.rand(3) * 0.05
            scene.set_ambient_light(ambient_light)
            alight_color = np.random.rand(3) * np.array((60, 20, 20)) + np.array([30, 10, 10])
            light_1_color = np.random.rand(3) * 20 + 20
            light_2_color = np.random.rand(3) * 10 + 5
            light_3_color = np.random.rand(3) * 10 + 5
            light_4_color = np.random.rand(3) * 10 + 5
            light_5_color = np.random.rand(3) * 10 + 5
            plight1.set_color(light_1_color * 0.01)
            plight2.set_color(light_2_color * 0.01)
            plight3.set_color(light_3_color * 0.01)
            plight4.set_color(light_4_color * 0.01)
            plight5.set_color(light_5_color * 0.01)
            alight.set_color(alight_color)

        def light_off_without_alight():
            alight.set_color([0.0, 0.0, 0.0])

    else:
        scene.set_ambient_light([0.5, 0.5, 0.5])
        plight1 = scene.add_point_light([-0.3, -0.3, 2.5], [30, 30, 30])
        plight2 = scene.add_point_light([2, -2, 2.5], [10, 10, 10])
        plight3 = scene.add_point_light([-2, 2, 2.5], [10, 10, 10])
        plight4 = scene.add_point_light([2, 2, 2.5], [10, 10, 10])
        plight5 = scene.add_point_light([-2, -2, 2.5], [10, 10, 10])

        # change light
        def lights_on():
            scene.set_ambient_light([0.5, 0.5, 0.5])
            plight1.set_color([30, 30, 30])
            plight2.set_color([10, 10, 10])
            plight3.set_color([10, 10, 10])
            plight4.set_color([10, 10, 10])
            plight5.set_color([10, 10, 10])
            alight.set_color([0.0, 0.0, 0.0])

        def lights_off():
            p_scale = 4.0
            scene.set_ambient_light([0.03, 0.03, 0.03])
            plight1.set_color([0.3 * p_scale, 0.1 * p_scale, 0.1 * p_scale])
            plight2.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight3.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight4.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight5.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            alight.set_color([60.0, 20.0, 20.0])

        def light_off_without_alight():
            p_scale = 4.0
            scene.set_ambient_light([0.03, 0.03, 0.03])
            plight1.set_color([0.3 * p_scale, 0.1 * p_scale, 0.1 * p_scale])
            plight2.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight3.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight4.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight5.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            alight.set_color([0.0, 0.0, 0.0])

    mount_T = np.eye(3)
    # mount_T = t3d.quaternions.quat2mat((-0.5, 0.5, 0.5, -0.5)) # TODO: remove this in sapien3, since in sapien2 the active light convention is broken and that's why we used this weird mount_T matrix
    if camera_type == 'd415':
        pose_rgb_irproj = sapien.Pose()
    elif camera_type == 'd435':
        pose_rgb_irproj = sapien.Pose.from_transformation_matrix(np.array(
            [[0.9999489188194275, 0.009671091102063656, 0.0029452370945364237, 0],
            [-0.009709948673844337, 0.999862015247345, 0.013478035107254982, -0.015-0.029],
            [-0.0028144833631813526, -0.013505944050848484, 0.9999048113822937, 0],
            [0.0, 0.0, 0.0, 1.0]]
        ))
    else:
        raise NotImplementedError()

    if camera_type == 'd415':
        fov = np.random.uniform(1.3, 2.0)
    elif camera_type == 'd435':
        fov = np.random.uniform(1.6755, 2.0943) # 99 - 120 degs
    else:
        raise NotImplementedError()
    
    tex = cv2.imread(os.path.join(materials_root, f"{camera_type}-pattern-sq.png"))

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        return result

    tmp_idx = np.random.randint(1e8)
    if rand_pattern:
        if camera_type == 'd415':
            angle = np.random.uniform(-90, 90)
        elif camera_type == 'd435':
            angle = np.random.uniform(-5, 5)
        else:
            raise NotImplementedError()
        from wand.image import Image as WandImage
        tex_tmp = WandImage.from_array(tex)
        tex_tmp.virtual_pixel = 'black'
        blur_radius = np.random.uniform(0.05, 2.5)
        tex_tmp.blur(radius=blur_radius, sigma=blur_radius / 3)
        
        distort_sq, distort_l = np.random.uniform(-0.002, 0.0), np.random.uniform(-0.08, 0.0)
        distort_c = np.random.uniform(1.0 + np.abs(distort_l) / 2, 1.0 + 1.15 * np.abs(distort_l))
        tex_tmp.distort('barrel', (0.0, distort_sq, distort_l, distort_c))
        
        tex_tmp = np.array(tex_tmp)
        cv2.imwrite("/tmp.png", tex_tmp)
        tex_tmp = rotate_image(tex_tmp, angle)
        cv2.imwrite(os.path.join(materials_root, f"{camera_type}-pattern-sq-tmp-{tmp_idx:08d}.png"), tex_tmp)

        alight = scene.add_active_light(
            pose=Pose([0.4, 0, 0.8]), # dummy pose, will be overwritten later
            # pose=Pose(cam_mount.get_pose().p, apos),
            color=[0, 0, 0], # dummy color, will be overwritten later
            fov=fov,
            tex_path=os.path.join(materials_root, f"{camera_type}-pattern-sq-tmp-{tmp_idx:08d}.png"),
        )
    else:
        alight = scene.add_active_light(
            pose=Pose([0.4, 0, 0.8]), # dummy pose, will be overwritten later
            # pose=Pose(cam_mount.get_pose().p, apos),
            color=[0, 0, 0], # dummy color, will be overwritten later
            fov=fov,
            tex_path=os.path.join(materials_root, f"{camera_type}-pattern-sq.png"),
        )

    cam_extrinsic_list = np.load(os.path.join(materials_root, "cam_db_neoneo.npy"))
    if fixed_angle:
        assert num_views <= cam_extrinsic_list.shape[0]
    else:
        # Obtain random camera poses around a specific center location
        # TODO: adjust these
        obj_center = np.array([0.425, 0, 0])
        alpha_range = [0, 2 * np.pi]
        # theta_range = [0.01, np.pi * 3 / 8]
        # radius_range = [0.8, 1.2]
        theta_range = [0.01, np.pi * 7 / 16]
        radius_range = [0.1, 1.4]
        angle_list = [
            (alpha, theta, radius)
            for alpha in np.linspace(alpha_range[0], alpha_range[1], 50)
            for theta in np.linspace(theta_range[0], theta_range[1], 20) # 10)
            for radius in np.linspace(radius_range[0], radius_range[1], 40) # 10)
        ]

    # set scene layout
    if primitives:
        num_asset = random.randint(PRIMITIVE_MIN, PRIMITIVE_MAX)
        primitive_info = {}
        for i in range(num_asset):
            info = load_random_primitives(scene, renderer=renderer, idx=i)
            primitive_info.update(info)
        primitive_info_keys = list(primitive_info.keys())
    elif primitives_v2:
        # num_asset = random.randint(PRIMITIVE_MIN, PRIMITIVE_MAX)
        clutter_option = 'regular' if np.random.random() < 0.35 else 'smaller'
        p_max = PRIMITIVE_MAX if clutter_option == 'smaller' else int(PRIMITIVE_MAX * 1.5)
        num_asset = int(np.exp(np.random.uniform(np.log(PRIMITIVE_MIN), np.log(p_max))))
        primitive_info = {}
        for i in range(num_asset):
            info = load_random_primitives_v2(scene, renderer=renderer, idx=i, clutter_option=clutter_option)
            primitive_info.update(info)
        primitive_info_keys = list(primitive_info.keys())
    else:
        if not os.path.exists(os.path.join(SCENE_DIR, f"{scene_id}/input.json")):
            logger.warning(f"{SCENE_DIR}/{scene_id}/input.json not exists.")
            return
        world_js = json.load(open(os.path.join(SCENE_DIR, f"{scene_id}/input.json"), "r"))
        assets = world_js.keys()
        poses_world = [None for _ in range(NUM_OBJECTS)]
        extents = [None for _ in range(NUM_OBJECTS)]
        scales = [None for _ in range(NUM_OBJECTS)]
        obj_ids = []
        object_names = []

        for obj_name in assets:
            load_obj(
                scene,
                obj_name,
                renderer=renderer,
                pose=sapien.Pose.from_transformation_matrix(world_js[obj_name]),
                is_kinematic=True,
                material_name="kuafu_material_new2",
            )
            obj_id = OBJECT_NAMES.index(obj_name)
            obj_ids.append(obj_id)
            object_names.append(obj_name)
            poses_world[obj_id] = world_js[obj_name]
            extents[obj_id] = np.array(
                [
                    float(OBJECT_DB[obj_name]["x_dim"]),
                    float(OBJECT_DB[obj_name]["y_dim"]),
                    float(OBJECT_DB[obj_name]["z_dim"]),
                ],
                dtype=np.float32,
            )
            scales[obj_id] = np.ones(3)
        obj_info = {
            "poses_world": poses_world,
            "extents": extents,
            "scales": scales,
            "object_ids": obj_ids,
            "object_names": object_names,
        }

    for view_id in range(num_views):
        folder_path = os.path.join(target_root, f"{scene_id}-{view_id}")
        os.makedirs(folder_path, exist_ok=True)
        if fixed_angle:
            cam_mount.set_pose(cv2ex2pose(cam_extrinsic_list[view_id]))
            ir_projector_pose = cam_mount.get_pose() * pose_rgb_irproj
            apos = ir_projector_pose.to_transformation_matrix()[:3, :3] @ mount_T
            apos = t3d.quaternions.mat2quat(apos)
            alight.set_pose(Pose(ir_projector_pose.p, apos))

        else:
            # Obtain random camera extrinsic
            sample_camera_near_primitive = np.random.random() < 0.6 # if True, randomly sample a camera pose near a primitive object
            sample_dist_primitive_rand = np.random.random()
            while True:
                if primitives or primitives_v2:
                    if sample_camera_near_primitive:
                        chosen_primitive_info_key_idx = np.random.randint(len(primitive_info_keys))
                        chosen_primitive = primitive_info[primitive_info_keys[chosen_primitive_info_key_idx]]
                        if sample_dist_primitive_rand < 0.30:
                            min_radius, max_radius = 0.01, 0.05
                        elif sample_dist_primitive_rand < 0.50:
                            min_radius, max_radius = 0.02, 0.18
                        else:
                            min_radius, max_radius = 0.15, 0.90
                        cam_extrinsic = sample_camera_pose_near_primitive(
                            primitive_obj=chosen_primitive, center=obj_center, min_radius=min_radius, max_radius=max_radius
                        )
                        # print("*****************cam pose near primitive", cam_extrinsic)
                    else:
                        if np.random.random() < 0.6:
                            alpha, theta, radius = angle_list[np.random.randint(len(angle_list))]
                            cam_extrinsic = spherical_pose(center=obj_center, radius=radius, alpha=alpha, theta=theta)
                        else:
                            # randomly sample a bird-eye view pose
                            cam_pos = np.random.uniform([0.1, -0.3, 0.4], [0.6, 0.3, 0.9])
                            lookat = np.random.uniform([-0.15, -0.15, 0.05], [0.15, 0.15, 0.10])
                            lookat[:2] = lookat[:2] + cam_pos[:2]
                            forward = (lookat - cam_pos) / np.linalg.norm(lookat - cam_pos)
                            left = np.cross([0, 0, 1], forward)
                            left = left / np.linalg.norm(left)
                            up = np.cross(forward, left)
                            cam_extrinsic = np.eye(4)
                            cam_extrinsic[:3, :3] = np.stack([forward, left, up], axis=1)
                            cam_extrinsic[:3, 3] = cam_pos
                        # print("*****************cam pose spherical", cam_extrinsic)
                    # if sample_camera_near_primitive and not check_camera_collision_with_primitive_dict(cam_extrinsic[:3, 3], primitive_info, eps=0.13):
                        # the camera pose is too far away from an object; resample a camera pose
                        # continue
                    cam_pos_arr = [cam_extrinsic[:3, 3], (cam_extrinsic @ cam_irL_rel_extrinsic_base)[:3, 3], (cam_extrinsic @ cam_irR_rel_extrinsic_base)[:3, 3]]
                    if not check_camera_collision_with_primitive_dict(cam_pos_arr, primitive_info, eps=0.006):
                        break
                else:
                    alpha, theta, radius = angle_list[np.random.randint(len(angle_list))]
                    cam_extrinsic = spherical_pose(center=obj_center, radius=radius, alpha=alpha, theta=theta)
                    break
            cam_mount.set_pose(sapien.Pose.from_transformation_matrix(cam_extrinsic)) # T^w_(w->(cam in ros))

            ir_projector_pose = cam_mount.get_pose() * pose_rgb_irproj
            apos = ir_projector_pose.to_transformation_matrix()[:3, :3] @ mount_T
            apos = t3d.quaternions.mat2quat(apos)
            alight.set_pose(Pose(ir_projector_pose.p, apos))
        if view_id == 0 and fixed_angle:
            # reproduce IJRR
            cam_rgb = base_cam_rgb
            cam_irl = base_cam_irl
            cam_irr = base_cam_irr
        else:
            cam_rgb = hand_cam_rgb
            cam_irl = hand_cam_irl
            cam_irr = hand_cam_irr

        if not os.path.exists(os.path.join(folder_path, f"{spp:04d}_irR_kuafu_half_no_ir.png")):
            # begin rendering
            lights_on()
            scene.update_render()
            # Render mono-view RGB camera
            cam_rgb.take_picture()
            p = cam_rgb.get_color_rgba()
            plt.imsave(os.path.join(folder_path, f"{spp:04d}_rgb_kuafu.png"), p)

            pos = cam_rgb.get_float_texture("Position")
            depth = -pos[..., 2]
            depth = (np.clip(depth, 0, 65) * 1000.0).astype(np.uint16)
            cv2.imwrite(os.path.join(folder_path, f"depth.png"), depth)
            vis_depth = visualize_depth(depth)
            cv2.imwrite(os.path.join(folder_path, f"depth_colored.png"), vis_depth)
            gl2cv = np.array([[1, 0, 0], [0, -1,  0], [0, 0, -1]], dtype=np.float32)
            normal = cam_rgb.get_float_texture("Normal")[..., :3] # Normal under GL camera frame
            normal = np.einsum('xyj,ij->xyi', normal, gl2cv)
            normal = ((normal + 1) * 1000.0).astype(np.uint16)
            cv2.imwrite(os.path.join(folder_path, f"normal.png"), normal)

            # Render multi-view RGB camera
            cam_irl.take_picture()
            p = cam_irl.get_color_rgba()
            plt.imsave(os.path.join(folder_path, f"{spp:04d}_rgbL_kuafu.png"), p)

            pos = cam_irl.get_float_texture("Position")
            depth = -pos[..., 2]
            depth = (np.clip(depth, 0, 65) * 1000.0).astype(np.uint16)
            cv2.imwrite(os.path.join(folder_path, f"depthL.png"), depth)
            vis_depth = visualize_depth(depth)
            cv2.imwrite(os.path.join(folder_path, f"depthL_colored.png"), vis_depth)
            normal = cam_irl.get_float_texture("Normal")[..., :3]
            normal = np.einsum('xyj,ij->xyi', normal, gl2cv)
            normal = ((normal + 1) * 1000.0).astype(np.uint16)
            cv2.imwrite(os.path.join(folder_path, f"normalL.png"), normal)


            cam_irr.take_picture()
            p = cam_irr.get_color_rgba()
            plt.imsave(os.path.join(folder_path, f"{spp:04d}_rgbR_kuafu.png"), p)

            pos = cam_irr.get_float_texture("Position")
            depth = -pos[..., 2]
            depth = (np.clip(depth, 0, 65) * 1000.0).astype(np.uint16)
            cv2.imwrite(os.path.join(folder_path, f"depthR.png"), depth)
            vis_depth = visualize_depth(depth)
            cv2.imwrite(os.path.join(folder_path, f"depthR_colored.png"), vis_depth)
            normal = cam_irr.get_float_texture("Normal")[..., :3]
            normal = np.einsum('xyj,ij->xyi', normal, gl2cv)
            normal = ((normal + 1) * 1000.0).astype(np.uint16)
            cv2.imwrite(os.path.join(folder_path, f"normalR.png"), normal)

            plt.close("all")

            lights_off()
            scene.update_render()

            # Render multi-view IR camera
            cam_irl.take_picture()
            p_l = cam_irl.get_color_rgba()
            cam_irr.take_picture()
            p_r = cam_irr.get_color_rgba()
            p_l = (p_l[..., :3] * 255).clip(0, 255).astype(np.uint8)
            p_r = (p_r[..., :3] * 255).clip(0, 255).astype(np.uint8)
            p_l = cv2.cvtColor(p_l, cv2.COLOR_RGB2GRAY)
            p_r = cv2.cvtColor(p_r, cv2.COLOR_RGB2GRAY)
            p_l = cv2.GaussianBlur(p_l, (3, 3), 1)
            p_r = cv2.GaussianBlur(p_r, (3, 3), 1)
            # irl = p_l[::2, ::2] # historical reason
            # irr = p_r[::2, ::2]
            irl, irr = p_l, p_r
            cv2.imwrite(os.path.join(folder_path, f"{spp:04d}_irL_kuafu_half.png"), irl)
            cv2.imwrite(os.path.join(folder_path, f"{spp:04d}_irR_kuafu_half.png"), irr)

            light_off_without_alight()
            scene.update_render()

            # Render multi-view IR camera without pattern
            cam_irl.take_picture()
            p_l = cam_irl.get_color_rgba()
            cam_irr.take_picture()
            p_r = cam_irr.get_color_rgba()
            p_l = (p_l[..., :3] * 255).clip(0, 255).astype(np.uint8)
            p_r = (p_r[..., :3] * 255).clip(0, 255).astype(np.uint8)
            p_l = cv2.cvtColor(p_l, cv2.COLOR_RGB2GRAY)
            p_r = cv2.cvtColor(p_r, cv2.COLOR_RGB2GRAY)
            p_l = cv2.GaussianBlur(p_l, (3, 3), 1)
            p_r = cv2.GaussianBlur(p_r, (3, 3), 1)
            # no_irl = p_l[::2, ::2] # historical reason
            # no_irr = p_r[::2, ::2]
            no_irl, no_irr = p_l, p_r
            cv2.imwrite(os.path.join(folder_path, f"{spp:04d}_irL_kuafu_half_no_ir.png"), no_irl)
            cv2.imwrite(os.path.join(folder_path, f"{spp:04d}_irR_kuafu_half_no_ir.png"), no_irr)

        else:
            logger.info(f"skip {folder_path} rendering")

        if fixed_angle:
            cam_extrinsic = cam_extrinsic_list[view_id]
        else:
            cam_extrinsic = pose2cv2ex(cam_extrinsic)
        if view_id == 0 and fixed_angle:
            # reproduce IJRR
            cam_irL_extrinsic = np.linalg.inv(np.linalg.inv(cam_extrinsic) @ cam_irL_rel_extrinsic_base)
            cam_irR_extrinsic = np.linalg.inv(np.linalg.inv(cam_extrinsic) @ cam_irR_rel_extrinsic_base)
            cam_intrinsic = cam_intrinsic_base
            cam_ir_intrinsic = cam_ir_intrinsic_base
        else:
            cam_irL_extrinsic = np.linalg.inv(np.linalg.inv(cam_extrinsic) @ cam_irL_rel_extrinsic_hand)
            cam_irR_extrinsic = np.linalg.inv(np.linalg.inv(cam_extrinsic) @ cam_irR_rel_extrinsic_hand)
            cam_intrinsic = cam_intrinsic_hand
            cam_ir_intrinsic = cam_ir_intrinsic_hand

        # Save scene info
        scene_info = {
            "extrinsic": cam_extrinsic,
            "extrinsic_l": cam_irL_extrinsic,
            "extrinsic_r": cam_irR_extrinsic,
            "intrinsic": cam_intrinsic,
            "intrinsic_l": cam_ir_intrinsic,
            "intrinsic_r": cam_ir_intrinsic,
        }
        if primitives or primitives_v2:
            scene_info["primitives"] = primitive_info
        else:
            scene_info.update(obj_info)

        with open(os.path.join(folder_path, "meta.pkl"), "wb") as f:
            pickle.dump(scene_info, f)

        logger.info(f"finish {folder_path} rendering")
    if rand_pattern:
        os.remove(os.path.join(materials_root, f"{camera_type}-pattern-sq-tmp-{tmp_idx:08d}.png"))
    scene = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--target-root", type=str, required=True)
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--nv", type=int, default=21)
    parser.add_argument("--rand-pattern", action="store_true")
    parser.add_argument("--fixed-angle", action="store_true")
    parser.add_argument("--primitives", action="store_true", help="use primitives")
    parser.add_argument("--primitives-v2", action="store_true", help="use primitives v2")
    args = parser.parse_args()

    assert not (args.primitives and args.primitives_v2), "primitives and v2 cannot be True in one run"
    render_scene(
        args.scene,
        repo_root=args.repo_root,
        target_root=args.target_root,
        spp=args.spp,
        num_views=args.nv,
        rand_pattern=args.rand_pattern,
        fixed_angle=args.fixed_angle,
        primitives=args.primitives,
        primitives_v2=args.primitives_v2,
    )
