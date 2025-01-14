"""
python data_rendering/render_script.py --sub 1 --total 200 --target-root ./dataset_sim --primitives-v2 --rand-pattern --rand-lighting
"""
import os
import os.path as osp
import sys
import time

import numpy as np
import sapien.core as sapien
from loguru import logger
from path import Path
import json

CUR_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from data_rendering.render_scene import render_scene, SCENE_DIR
from data_rendering.utils.render_utils import load_pickle, timeout

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sub", type=int, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--target-root", type=str, required=True)
    parser.add_argument("--camera-type", type=str, default="d435", choices=["d435", "d415"])
    parser.add_argument("--camera-resolution", nargs=2, type=int, default=[848, 480])
    parser.add_argument("--num-scene", type=int, default=2000)
    parser.add_argument("--num-view", type=int, default=21)
    parser.add_argument("--rand-pattern", action="store_true")
    parser.add_argument("--fixed-angle", action="store_true")
    parser.add_argument("--rand-lighting", action="store_true")
    parser.add_argument("--primitives", action="store_true", help="use primitives")
    parser.add_argument("--primitives-v2", action="store_true", help="use primitives v2")
    parser.add_argument("--rand-table", action="store_true", help="use random material for table")
    parser.add_argument("--rand-env", action="store_true", help="use random environment map")
    args = parser.parse_args()
    args.camera_resolution = tuple(args.camera_resolution)

    spp = 128
    num_view = args.num_view

    repo_root = REPO_ROOT
    target_root = args.target_root
    data_root = osp.join(args.target_root, "data")
    Path(data_root).makedirs_p()

    timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
    name = "render_" + target_root.split("/")[-1]
    filename = f"log.render.sub{args.sub:02d}.tot{args.total}.{timestamp}.txt"
    # set up logger
    logger.remove()
    fmt = (
        f"<green>{{time:YYYY-MM-DD HH:mm:ss.SSS}}</green> | "
        f"<cyan>{name}</cyan> | "
        f"<lvl>{{level}}</lvl> | "
        f"<lvl>{{message}}</lvl>"
    )

    # logger to file
    log_file = Path(target_root) / filename
    logger.add(log_file, format=fmt)

    # logger to std stream
    logger.add(sys.stdout, format=fmt)
    logger.info(f"Args: {args}")

    if args.primitives or args.primitives_v2:
        total_scene = args.num_scene # 2000
        scene_names = np.arange(total_scene)
        sub_total_scene = len(scene_names) // args.total
        sub_scene_list = (
            scene_names[(args.sub - 1) * sub_total_scene : args.sub * sub_total_scene]
            if args.sub < args.total
            else scene_names[(args.sub - 1) * sub_total_scene :]
        )
    else:
        total_scene = args.num_scene # 1000
        scene_names = np.arange(total_scene)
        sub_total_scene = len(scene_names) // args.total
        sub_scene_list = []
        if args.sub < args.total:
            for s in scene_names[(args.sub - 1) * sub_total_scene : args.sub * sub_total_scene]:
                sub_scene_list.append(f"0-{s}")
                sub_scene_list.append(f"1-{s}")
        else:
            for s in scene_names[(args.sub - 1) * sub_total_scene :]:
                sub_scene_list.append(f"0-{s}")
                sub_scene_list.append(f"1-{s}")

    logger.info(f"Generating {len(sub_scene_list)} scenes from {sub_scene_list[0]} to {sub_scene_list[-1]}")

    # build scene
    sim = sapien.Engine()
    sim.set_log_level("warning")
    sapien.render_config.camera_shader_dir = "rt"
    sapien.render_config.viewer_shader_dir = "rt"
    """
    sapien.KuafuRenderer.set_log_level("warning")

    render_config = sapien.KuafuConfig()
    render_config.use_viewer = False
    render_config.use_denoiser = True
    render_config.spp = spp
    render_config.max_bounces = 8

    renderer = sapien.KuafuRenderer(render_config)
    """
    sapien.render_config.use_viewer = False
    sapien.render_config.rt_use_denoiser = False
    # sapien.render_config.rt_samples_per_pixel = 256
    sapien.render_config.spp = spp
    sapien.render_config.max_bounces = 8
    renderer = sapien.SapienRenderer()
    sim.set_renderer(renderer)

    for sc in sub_scene_list:
        done = True
        for v in range(num_view):
            if osp.exists(osp.join(data_root, f"{sc}-{v}/meta.pkl")):
                try:
                    load_pickle(osp.join(data_root, f"{sc}-{v}/meta.pkl"))
                except Exception as e:
                    logger.error(f"{sc} fail to load meta: {e}")
                    done = False
                    break
            else:
                done = False
                break
        if done:
            logger.info(f"Skip scene {sc} rendering")
            continue
        # if not (args.primitives or args.primitives_v2):
        #     all_exist = True
        #     if not os.path.exists(os.path.join(SCENE_DIR, f"{sc}/input.json")):
        #         logger.warning(f"{SCENE_DIR}/{sc}/input.json not exists.")
        #         continue
        #     world_js = json.load(open(os.path.join(SCENE_DIR, f"{sc}/input.json"), "r"))
        #     if "rubik" in world_js.keys():
        #         all_exist = False
        #         logger.warning(f"Rubik is in {sc}")
        #     else:
        #         for i in range(num_view):
        #             if not osp.exists(osp.join(target_root, f"{sc}-{i}/meta.pkl")):
        #                 all_exist = False
        #                 break
        #     if all_exist:
        #         logger.info(f"Move {target_root}/{sc} to {data_root}/{sc}")
        #         for i in range(num_view):
        #             (Path(target_root) / f"{sc}-{i}").move(osp.join(data_root, f"{sc}-{i}"))
        #         continue

        logger.info(f"Rendering scene {sc}")
        
        with timeout(seconds=50+25*num_view):
            render_scene(
                sim=sim,
                renderer=renderer,
                scene_id=sc,
                repo_root=repo_root,
                target_root=data_root,
                camera_type=args.camera_type,
                camera_resolution=args.camera_resolution,
                spp=spp,
                num_views=num_view,
                rand_pattern=args.rand_pattern,
                fixed_angle=args.fixed_angle,
                primitives=args.primitives,
                primitives_v2=args.primitives_v2,
                rand_lighting=args.rand_lighting,
                rand_table=args.rand_table,
                rand_env=args.rand_env,
            )
    
    exit(0)
