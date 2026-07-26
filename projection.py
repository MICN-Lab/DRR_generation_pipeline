import numpy as np
import tigre
from tigre.utilities.geometry import Geometry

class Projector:

    def __init__(self, config):
        self.config = config

    
    def setup_geometry(self, vol_shape_zyx, vol_spacing_zyx):

        # Geometry object
        geo = Geometry()
        # Geometry type (cone or parallel)
        geo.mode = self.config.mode
        # Distance Source Detector (mm)
        geo.DSD = self.config.sid
        # Distance Source Origin (mm)
        geo.DSO = self.config.sod


        # Number of voxels in the image (3x1 array)
        geo.nVoxel = np.array(vol_shape_zyx, dtype = np.int32)
        # Size of each of the voxels in mm (3x1 array)
        geo.dVoxel = np.array(vol_spacing_zyx, dtype= np.float32)
        # Total size in mm of the image (number of voxel * size of voxel)
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        
        # z axis
        phys_height = geo.sVoxel[0]
        # diagonal (sqrt(y² + x²))
        phys_width_diag = np.sqrt(geo.sVoxel[1]**2 + geo.sVoxel[2]**2)

        M = geo.DSD / geo.DSO
        sDet_h = phys_height * M * self.config.det_padding
        sDet_w = phys_width_diag * M * self.config.det_padding

        # Number of voxels in the detector plane (2x1 array)
        geo.nDetector = np.array(self.config.det_pixels, dtype=np.int32)
        # Total size in mm of the detector (2x1 array)
        geo.sDetector = np.array([sDet_h, sDet_w], dtype=np.float32)
        # Size of each of the pixels in the detector in mm (2x1 array)
        geo.dDetector = geo.sDetector / geo.nDetector

        # Offset of image from origin (mm)
        geo.offOrigin = np.zeros(3, dtype=np.float32)
        # Offset of Detector (mm)
        geo.offDetector = np.zeros(3, dtype=np.float32)
        geo.rotDetector = np.zeros(3, dtype=np.float32)
        geo.COR = 0.0

        return geo
    
    def project(self, volume, geo):
        """
        Generate projections from the volume with the angles parameters in the config file
        """
        angles_deg = np.linspace(0, self.config.end_angle, self.config.n_angles, endpoint=True)

        angles_rad = np.deg2rad(angles_deg).astype(np.float32)
        
        projs = tigre.Ax(volume, geo, angles_rad)

        return projs, angles_deg, geo