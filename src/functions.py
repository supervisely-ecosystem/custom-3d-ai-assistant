import os
import numpy as np
import open3d as o3d
import supervisely as sly
from supervisely.geometry.cuboid_3d import Cuboid3d, Vector3d


labeling_proposals = {}


def read_pcd(pcd_id, api):
    local_pcd_path = f"input_pcds/{pcd_id}.pcd"
    if not os.path.exists(local_pcd_path):
        api.pointcloud.download_path(pcd_id, local_pcd_path)
    pcd = o3d.io.read_point_cloud(local_pcd_path, format="pcd")
    pcd_points = np.asarray(pcd.points)
    return pcd, pcd_points


def generate_random_cuboid(pcd, mask_indices):
    masked_pcd = pcd.select_by_index(mask_indices)
    center = masked_pcd.get_center()
    position = Vector3d(float(center[0]), float(center[1]), float(center[2]))
    dimensions = Vector3d(
        float(np.random.uniform(0.5, 1.0)),
        float(np.random.uniform(0.5, 1.0)),
        float(np.random.uniform(1.5, 2.0)),
    )
    rotation = Vector3d(0.0, 0.0, float(np.random.uniform(0.0, 2 * np.pi)))
    return Cuboid3d(position, rotation, dimensions)


def clone_cuboid_with_random_shift(source_cuboid, max_shift=0.5):
    pos = source_cuboid.position
    rot = source_cuboid.rotation
    dim = source_cuboid.dimensions
    new_position = Vector3d(
        pos.x + float(np.random.uniform(-max_shift, max_shift)),
        pos.y + float(np.random.uniform(-max_shift, max_shift)),
        pos.z + float(np.random.uniform(-max_shift, max_shift)),
    )
    new_rotation = Vector3d(rot.x, rot.y, rot.z)
    new_dimensions = Vector3d(dim.x, dim.y, dim.z)
    return Cuboid3d(new_position, new_rotation, new_dimensions)


def generate_random_clusters(pcd, k=4, n=300):
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return []
    tree = o3d.geometry.KDTreeFlann(pcd)
    k_eff = min(k, len(points))
    n_eff = min(n, len(points))
    center_idxs = np.random.choice(len(points), size=k_eff, replace=False)
    clusters = []
    for ci in center_idxs:
        _, idx, _ = tree.search_knn_vector_3d(points[ci], n_eff)
        clusters.append(list(idx))
    return clusters


def get_2d_anns(image_id, dataset_id, photo_context_img, api, figure_ids):
    figures = api.image.figure.download(dataset_id, [image_id])[image_id]
    anns_2d = []
    for figure in figures:
        if len(figure_ids) > 0 and figure.id not in figure_ids:
            continue
        if figure.geometry_type == "bitmap":
            geometry = sly.Bitmap.from_json(figure.geometry)
            mask = np.zeros(photo_context_img.shape, dtype=np.uint8)
            geometry.draw(bitmap=mask, color=[1, 1, 1])
            mask = mask[:, :, :2]
            anns_2d.append(("bitmap", mask, figure.id))
        elif figure.geometry_type == "rectangle":
            geometry = sly.Rectangle.from_json(figure.geometry)
            bbox = [geometry.left, geometry.top, geometry.right, geometry.bottom]
            anns_2d.append(("rectangle", bbox, figure.id))
        elif figure.geometry_type == "polygon":
            polygon_obj_class = sly.ObjClass("polygon", sly.Polygon)
            polygon_label = sly.Label(
                sly.Polygon.from_json(figure.geometry), polygon_obj_class
            )
            bitmap_obj_class = sly.ObjClass("bitmap", sly.Bitmap)
            bitmap_label = polygon_label.convert(bitmap_obj_class)[0]
            geometry = bitmap_label.geometry
            mask = np.zeros(photo_context_img.shape, dtype=np.uint8)
            geometry.draw(bitmap=mask, color=[1, 1, 1])
            mask = mask[:, :, :2]
            anns_2d.append(("polygon", mask, figure.id))
        elif figure.geometry_type == "line":
            geometry = sly.Polyline.from_json(figure.geometry)
            line_points_loc = geometry.exterior
            line_points = [[point.col, point.row] for point in line_points_loc]
            anns_2d.append(("line", line_points, figure.id))
    return anns_2d
