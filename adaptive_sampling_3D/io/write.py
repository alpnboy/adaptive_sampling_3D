import open3d as o3d
import os

def save_pcd(input_pcd, result_path):
    
    try:
        directory = os.path.dirname(result_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        success = o3d.io.write_point_cloud(result_path, input_pcd)
        return success
    
    except Exception as e:
        print(f"Error: {e}")
        return False