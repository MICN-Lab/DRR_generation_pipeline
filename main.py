import os
import pathlib
from config import Config
from preprocessing import Preprocessor
from projection import Projector
from utils import save_metadata, save_nifti, save_png


def main():


    config = Config()
    
    preprocessor = Preprocessor(config)
    projector = Projector(config)

    files = sorted([f for f in os.listdir(config.input_dir) if f.endswith((".nii", ".nii.gz"))])
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Starting : {len(files)} files")

    for filename in files:
        case_id = pathlib.Path(filename).stem.split('.')[0]
        case_dir = os.path.join(config.output_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)

        print(f"--> {case_id}")

        try:
            vol_zyx, spacing_zyx = preprocessor.process(file_path=os.path.join(config.input_dir, filename))
        except Exception as e:
            print(f"[ERROR] {case_id} skipped: {e}")
            continue
        
        # Saving the processed CT
        # (z, y, x) to (x, y, z) for SimpleITK
        save_nifti(vol_zyx, spacing_zyx[::-1], os.path.join(case_dir, "ct.nii.gz"))

        # Projection
        geo = projector.setup_geometry(vol_zyx.shape, spacing_zyx)

        projections, angles, final_geo = projector.project(vol_zyx, geo)

        # Saving DRRs
        # (y,x ) to (x, y) for SimpleITK
        drr_spacing_xy = (float(final_geo.dDetector[1]), float(final_geo.dDetector[0]))
        
        for i, angle in enumerate(angles):
            proj_slice = projections[i]
            base_name = f"{case_id}_drr_angle{int(round(angle)):03d}"

            # Save each projection as a NIfTI
            save_nifti(proj_slice, drr_spacing_xy, os.path.join(case_dir, base_name + ".nii.gz"))

            # Save each projection as a PNG
            if config.write_png:
                save_png(proj_slice, os.path.join(case_dir, base_name + ".png"))
        
        # Save metadata
        save_metadata(case_id, vol_zyx.shape, spacing_zyx, final_geo, angles, config.det_padding, case_dir)

    print("Done")



if __name__ == "__main__":
    main()