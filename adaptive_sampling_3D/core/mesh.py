import open3d as o3d
import numpy as np

def mesh_to_cloud_poissondisk(mesh, n_points = 100000):
    """ Converts a mesh to a point cloud using Poisson Disk Sampling.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input Open3D triangle mesh.
        n_points (int, optional): Number of points to sample. Defaults to 100000.

    Returns:
        o3d.geometry.PointCloud: The sampled point cloud with Poisson Disk uniform distribution.
    """
    
    pcd = mesh.sample_points_poisson_disk(n_points, radius = 3)

    return pcd

def mesh_to_cloud_uniform(mesh, n_points = 100000):
    """ Converts a mesh to a point cloud using Uniform Sampling.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input Open3D triangle mesh.
        n_points (int, optional): Number of points to sample. Defaults to 100000.

    Returns:
        o3d.geometry.PointCloud: The sampled point cloud.
    """
    
    pcd = mesh.sample_points_uniformly(n_points)

    return pcd

def estimate_npoints_sigmoid(mesh, n_max = 20000):
    """ Estimates sampling density based on mesh surface area using a sigmoid function.
    
    The function cleans the mesh geometry before calculating the surface area to 
    ensure an accurate estimation. 

    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh to analyze.
        n_max (int, optional): Upper bound for the number of points. Defaults to 20000.

    Returns:
        float: Estimated number of points based on the sigmoid curve.
    """
    
    # Clean mesh from aberrant geometries
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    
    # Get surface area to compare with upper bound area
    area = mesh.get_surface_area()
    area = area / 2000000           # Normalizing factor assumed as largest possible surface in tile
    a_mid = 0.5                     # Sigmoid mid point
    k = 4                           # Curve growth rate
    
    n_points = n_max / (1.0 + np.exp(-k * (area - a_mid)))
    
    return n_points



