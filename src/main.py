import supervisely as sly
from dotenv import load_dotenv
import supervisely.app.development as sly_app_development
from supervisely._utils import is_debug_with_sly_net
import os
from fastapi import Request, BackgroundTasks
import traceback
import src.functions as f
import functools
import numpy as np
import open3d as o3d



# for debug, has no effect in production
load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("debug.env")

app = sly.Application()

original_dir = os.getcwd()

if is_debug_with_sly_net():
    sly_app_development.supervisely_vpn_network(action="up")
    task = sly_app_development.create_debug_task(sly.env.team_id(), port="8000")
    os.chdir(original_dir)

server = app.get_server()


@server.post("/interactive_3d_detection")
def detect_cuboids(request: Request):
    try:
        # get input data
        api = request.state.api
        state = request.state.state
        current_pcd_id = state["pcd_id"]
        sly.logger.info(f"Detecting cuboids on point cloud with ID {current_pcd_id}...")
        # download input point cloud
        current_pcd, _ = f.read_pcd(current_pcd_id, api)
        mask_indices = state["indices"]

        geometry = f.generate_random_cuboid(current_pcd, mask_indices)

        response = {
            "result": geometry.to_json(),
            "error": None,
        }
        return response
    except Exception as e:
        sly.logger.warning("An error occured:")
        print(traceback.format_exc())
        response = {"result": None, "error": repr(e)}
        return response


@server.post("/track")
def start_track(request: Request, task: BackgroundTasks):
    task.add_task(track_cuboids, request)
    return {"message": "Track task started."}


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as exc:
            request: Request = args[0]
            context = request.state.context
            api: sly.Api = request.state.api
            track_id = context["trackId"]
            api.logger.error(f"An error occured: {repr(exc)}")

            api.post(
                "point-clouds.episodes.notify-annotation-tool",
                data={
                    "type": "point-cloud-episodes:tracking-error",
                    "data": {
                        "trackId": track_id,
                        "error": {"message": repr(exc)},
                    },
                },
            )
        return value

    return wrapper


@send_error_data
def track_cuboids(request: Request):
    api = request.state.api
    # extract track id, list of point cloud and object ids
    context = request.state.context
    track_id = context["trackId"]
    current_dataset_id = context["datasetId"]
    pcd_ids = context["pointCloudIds"]
    sly.logger.info(f"Tracking cuboids on point clouds with IDs {pcd_ids}...")
    direction = context["direction"]
    if direction == "backward":
        pcd_ids = pcd_ids[::-1]
    object_ids = context["objectIds"]

    # get cuboids selected for tracking
    first_pcd_id = pcd_ids[0]
    ann_info = api.pointcloud.annotation.download(first_pcd_id)
    selected_figures = [
        figure
        for figure in ann_info["frames"][0]["figures"]
        if figure["objectId"] in object_ids
    ]
    cuboids = [
        sly.deserialize_geometry(fig["geometryType"], fig["geometry"])
        for fig in selected_figures
    ]

    pbar_position = 0
    pbar_total = len(pcd_ids[1:]) * len(context["figureIds"])

    for idx, current_pcd_id in enumerate(pcd_ids[1:]):
        # get access to previous and current point clouds
        target_pcd, _ = f.read_pcd(current_pcd_id, api)
        previous_pcd_id = pcd_ids[idx]
        source_pcd, _ = f.read_pcd(previous_pcd_id, api)

        tracked_cuboids = []

        for it, (object_id, source_cuboid) in enumerate(zip(object_ids, cuboids)):

            tracked_cuboid = f.clone_cuboid_with_random_shift(source_cuboid, max_shift=0.5)
            tracked_cuboids.append(tracked_cuboid)

            api.pointcloud_episode.figure.create(
                current_pcd_id,
                object_id,
                tracked_cuboid.to_json(),
                "cuboid_3d",
                track_id,
            )

            pbar_position += 1
            api.pointcloud_episode.notify_progress(
                track_id,
                current_dataset_id,
                pcd_ids,
                pbar_position,
                pbar_total,
            )

        cuboids = tracked_cuboids

    api.logger.info("Successfully finished tracking process")


@server.post("/generate_clusters")
def generate_clusters(request: Request):
    api = request.state.api
    state = request.state.state
    pcd_id = state["pcd_id"]
    sly.logger.info(f"Generating labeling proposal clusters for point cloud with ID {pcd_id}...")
    pcd, _ = f.read_pcd(pcd_id, api)
    f.labeling_proposals[pcd_id] = f.generate_random_clusters(pcd)
    return {"result": None, "error": None}


@server.post("/get_labeling_proposal")
def get_labeling_proposal(request: Request):
    api = request.state.api
    # get pcd id
    state = request.state.state
    pcd_id = state["pcd_id"]

    click_coordinate = np.asarray(state["click_coordinate"])
    pcd, _ = f.read_pcd(pcd_id, api)

    # In a real impl this would consume f.labeling_proposals[pcd_id] from /generate_clusters.
    # Here we just crop a sphere around the click and return a random cuboid from that crop.
    sphere_radius = 4.85 * np.random.uniform(1.05, 1.1)
    cloud_tree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = cloud_tree.search_radius_vector_3d(click_coordinate, sphere_radius)

    if k == 0:
        sly.logger.warning("No labeling proposals found")
        return {"result": None}

    cluster_cuboid = f.generate_random_cuboid(pcd, list(idx))
    return {"result": cluster_cuboid.to_json(), "error": None}


@server.post("/segment_ground")
def get_ground_indices(request: Request):
    api = request.state.api
    # get pcd id
    state = request.state.state
    pcd_id = state["pcd_id"]
    sly.logger.info(f"Segmenting ground on point cloud with ID {pcd_id}...")
    # read point cloud
    pcd, _ = f.read_pcd(pcd_id, api)
    ground_indexes = np.random.choice(range(len(pcd.points)), round(0.3 * len(pcd.points)), replace=False)

    response = {
        "result": ground_indexes.tolist(),
        "error": None,
    }
    return response


@server.post("/transfer_masks_to_pcd")
def transfer_masks_to_pcd(request: Request):
    api = request.state.api
    # get pcd id
    state = request.state.state
    pcd_id = state["pcd_id"]
    sly.logger.info(
        f"Transfering annotations from photo context to point cloud with ID {pcd_id}..."
    )
    photo_context_img_id = state["image_id"]
    figure_ids = state.get("figure_ids", [])
    dataset_id = api.pointcloud.get_info_by_id(pcd_id).dataset_id
    # read point cloud
    pcd, pcd_points = f.read_pcd(pcd_id, api)

    # match the count of 2D figures on the photo context image
    figures = api.image.figure.download(dataset_id, [photo_context_img_id])[photo_context_img_id]
    if figure_ids:
        figures = [fig for fig in figures if fig.id in figure_ids]

    anns_3d = []
    n_pts = len(pcd_points)
    if n_pts > 0:
        tree = o3d.geometry.KDTreeFlann(pcd)
        n_neighbors = min(300, n_pts)
        for fig in figures:
            seed_idx = int(np.random.choice(n_pts))
            _, idx, _ = tree.search_knn_vector_3d(pcd_points[seed_idx], n_neighbors)
            cuboid = f.generate_random_cuboid(pcd, list(idx))
            anns_3d.append(
                {
                    "geometryType": "cuboid_3d",
                    "geometry": cuboid.to_json(),
                    "srcFigureId": fig.id,
                }
            )

    response = {"result": anns_3d}
    return response