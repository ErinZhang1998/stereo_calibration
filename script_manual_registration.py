import copy
import json
import time
import glob

import numpy as np
import open3d as o3d


def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def manual_registration(source, target):
    # source = o3d.io.read_point_cloud(source_path)
    # target = o3d.io.read_point_cloud(target_path)

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)

    assert (len(picked_id_source) >= 3 and 
            len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target, o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print("registered transformation:", reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)

    return reg_p2p.transformation

def crop_point_cloud(pc_path, scale_down=False):
    print("Demo for manual geometry cropping")
    print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    pcd = o3d.io.read_point_cloud(pc_path)
    if scale_down:
        pcd.scale(0.001, [0, 0, 0])
    o3d.visualization.draw_geometries_with_editing([pcd])

def compare_pcd(pc1_path, pc2_path):
    pcd1 = o3d.io.read_point_cloud(pc1_path)
    pcd2 = o3d.io.read_point_cloud(pc2_path)
    o3d.visualization.draw_geometries([pcd1, pcd2])

def save_transformed_pcd(pc_path, trans_pc_path, trans_np, mode='source'):
    pcd = o3d.io.read_point_cloud(pc_path)
    if mode == 'source':
        pcd.transform(trans_np)
    elif mode == 'target':
        pcd.transform(np.linalg.pinv(trans_np))
    o3d.io.write_point_cloud(trans_pc_path, pcd)

if __name__ == "__main__":

    # scene_fn_lst = sorted(glob.glob('img_data/calibration/scene_pcd*'))
    # for scene_fn in scene_fn_lst:
    # scene_fn = "img_data/calibration/scene_pcd2.pcd"
    # crop_point_cloud(scene_fn, scale_down=True)

    source_pcd = o3d.geometry.PointCloud()
    src_fn_lst = sorted(glob.glob('img_data/calibration/arm*'))
    for src_fn in src_fn_lst:
        source_pcd += o3d.io.read_point_cloud(src_fn)
    
    target_pcd = o3d.geometry.PointCloud()
    tgt_fn_lst = sorted(glob.glob('img_data/calibration/cropped*'))
    for tgt_fn in tgt_fn_lst:
        target_pcd += o3d.io.read_point_cloud(tgt_fn)

    trans_np = manual_registration(source_pcd, target_pcd)
    print("trans np:", trans_np)

    # trans_np = np.array([[-0.99384385, -0.10860472, -0.02189564, -0.58579925],
    #                      [ 0.00386441,  0.16352983, -0.98653082, -0.06514786],
    #                      [ 0.1107225 , -0.9805422 , -0.16210342,  1.04138236],
    #                      [ 0.        ,  0.        ,  0.        ,  1.        ]])

    # with open('data/trans_params.json', 'r') as f:
    #     trans_params = json.load(f)
    # time_id = int(time.time())
    # trans_params[f'rob2cam_{time_id}'] = trans_np.tolist()
    # trans_params[f'cam2rob_{time_id}'] = np.linalg.inv(trans_np).tolist()
    # with open('data/trans_params.json', 'w') as f:
    #     json.dump(trans_params, f, indent=2)

    # print("trans np:", trans_np)

    # trans_pc_path = "img_data/trans_tree4.ply"
    # save_transformed_pcd(target_path, trans_pc_path, 
    #                      trans_np, mode='target')

    # post-process point cloud
    # crop_point_cloud(trans_pc_path, scale_down=False)
