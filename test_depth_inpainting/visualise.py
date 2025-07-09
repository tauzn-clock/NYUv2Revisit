import numpy as np
import open3d as o3d

def img_over_pcd(points, img, filepath=None):

    # Visualize the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    color = img.reshape(-1, 3) / 255.0  # Normalize color values to [0, 1]

    point_cloud.colors = o3d.utility.Vector3dVector(color)

    tf = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    point_cloud.transform(tf)

    if filepath is not None:
        # Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the point cloud to the visualizer
        vis.add_geometry(point_cloud)

        opt = vis.get_render_option()
        opt.point_size = 2

        view_control = vis.get_view_control()
        view_control.set_zoom(0.6) 
        view_control.rotate(0, -100)

        img = np.array(vis.capture_screen_float_buffer(True))
        left = img.shape[1]
        right = 0
        top = img.shape[0]
        bottom = 0

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.sum(img[i,j]) < 3:
                    left = min(left, j)
                    right = max(right, j)
                    top = min(top, i)
                    bottom = max(bottom, i)

        output = img[top:bottom, left:right]
        
        plt.imsave(filepath, output)
    else:
        return point_cloud


if __name__ == "__main__":
    
    from PIL import Image
    from process_depth import get_3d

    IMG_PATH = "/scratchdata/InformationOptimisation/rgb/0.png"
    #DEPTH_PATH = "/scratchdata/InformationOptimisation/depth/0.png"
    DEPTH_PATH = "/NYUv2Revisit/test_depth_inpainting/0_APGL.png"

    img = Image.open(IMG_PATH)
    img = np.array(img)
    depth = Image.open(DEPTH_PATH)
    depth = np.array(depth, dtype=np.float32) / 1000.0

    pts_3d = get_3d(depth, [518.8579, 518.8579, 325.5824, 253.7362])

    pcd = img_over_pcd(pts_3d, img)

    o3d.visualization.draw_geometries([pcd])