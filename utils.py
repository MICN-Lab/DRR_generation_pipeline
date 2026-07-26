import os
import json
import numpy as np
import SimpleITK as sitk
import imageio.v2 as imageio

def save_nifti(arr, spacing, path):
    img = sitk.GetImageFromArray(arr)
    
    # Gestion du spacing (parfois 2D, parfois 3D)
    # On s'assure que c'est un tuple de floats
    img.SetSpacing(tuple(float(x) for x in spacing))
    
    # Origine à zéro par défaut
    # On crée un tuple de zéros de la bonne taille (2 ou 3)
    img.SetOrigin((0.0,) * img.GetDimension())

    # --- GESTION DE LA DIRECTION SELON LA DIMENSION ---
    if img.GetDimension() == 3:
        # C'est le CT : On applique le correctif RAS -> LPS pour ITK-SNAP
        # Matrice 3x3 (9 éléments)
        direction_3d = (
            -1.0, 0.0, 0.0,
             0.0, -1.0, 0.0,
             0.0, 0.0, 1.0
        )
        img.SetDirection(direction_3d)
        
    elif img.GetDimension() == 2:
        # C'est une DRR : On laisse l'identité standard 2D
        # Matrice 2x2 (4 éléments) : (1, 0, 0, 1)
        # Pas besoin de correction spéciale pour les DRRs en général
        direction_2d = (1.0, 0.0, 0.0, 1.0)
        img.SetDirection(direction_2d)

    sitk.WriteImage(img, path)

def save_png(arr, path):
    p_min, p_max = arr.min(), arr.max()
    if p_max - p_min > 1e-6:
        norm = (arr - p_min) / (p_max - p_min)
        png = (norm * 255).astype(np.uint8)
        imageio.imwrite(path, png)

def save_metadata(case_id, vol_shape, vol_spacing, geo, angles, det_padding, out_dir):
    meta = {
        "case_id": case_id,
        "ct": {
            "shape_zyx": list(vol_shape),
            "spacing_zyx": list(map(float, vol_spacing))
        },
        "drr": {
            "SID": float(geo.DSD),
            "SOD": float(geo.DSO),
            "detector_padding": float(det_padding),
            "sDetector": list(map(float, geo.sDetector)),
            "dDetector": list(map(float, geo.dDetector)),
            "nDetector": list(map(int, geo.nDetector)),
            "angles_deg": list(map(float, angles))
        }
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)