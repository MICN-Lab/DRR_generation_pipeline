import numpy as np
import SimpleITK as sitk


class Coregistrator:

    def __init__(self, config):
        self.config = config

        # Value used for voxels that fall outside the moving image.
        self.fill_value = float(config.hu_window[0])

        # Registration hyperparameters (overridable via Config).
        self.n_histogram_bins = getattr(config, "reg_histogram_bins", 50)
        self.sampling_percentage = getattr(config, "reg_sampling_percentage", 0.1)
        self.learning_rate = getattr(config, "reg_learning_rate", 1.0)
        self.min_step = getattr(config, "reg_min_step", 1e-4)
        self.n_iterations = getattr(config, "reg_iterations", 200)
        self.shrink_factors = getattr(config, "reg_shrink_factors", (4, 2, 1))
        self.smoothing_sigmas = getattr(config, "reg_smoothing_sigmas", (2.0, 1.0, 0.0))

        # Coarse search spans +/- coarse_rotation_step * n_coarse_steps per axis.
        self.coarse_rotation_step = getattr(config, "reg_coarse_rotation_step", np.deg2rad(20.0))
        self.n_coarse_steps = getattr(config, "reg_coarse_steps", 2)

        # Fixed seed keeps metric sampling reproducible.
        self.seed = getattr(config, "reg_seed", 42)

        # Reference volume, cached as (arr_zyx, spacing_zyx).
        self._reference = None

    # --- array <-> SimpleITK helpers ------------------------------------

    def _to_sitk(self, arr_zyx, spacing_zyx):
        img = sitk.GetImageFromArray(np.ascontiguousarray(arr_zyx, dtype=np.float32))
        # SimpleITK spacing is (x, y, z); our arrays are (z, y, x).
        img.SetSpacing(tuple(float(s) for s in spacing_zyx[::-1]))
        return img

    def _to_numpy(self, img):
        return sitk.GetArrayFromImage(img).astype(np.float32)

    # --- reference handling ---------------------------------------------

    def load_reference(self):
        reference_path = getattr(self.config, "reference_path", None)
        if reference_path is None:
            raise ValueError("add path")

        # Imported here so array-only use does not require monai/torch.
        from preprocessing import Preprocessor

        preprocessor = Preprocessor(self.config)
        arr_zyx, spacing_zyx = preprocessor.process(reference_path)
        self._reference = (arr_zyx, spacing_zyx)
        return arr_zyx, spacing_zyx

    # --- registration ----------------------------------------------------

    def _build_registration(self):
        """Multi-resolution MI registration with a regular-step gradient descent."""
        reg = sitk.ImageRegistrationMethod()

        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=self.n_histogram_bins)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(self.sampling_percentage, seed=self.seed)
        reg.SetInterpolator(sitk.sitkLinear)

        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=self.learning_rate,
            minStep=self.min_step,
            numberOfIterations=self.n_iterations,
        )
        reg.SetOptimizerScalesFromPhysicalShift()

        reg.SetShrinkFactorsPerLevel(list(self.shrink_factors))
        reg.SetSmoothingSigmasPerLevel(list(self.smoothing_sigmas))
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        return reg

    def _coarse_rotation_search(self, fixed, moving, initial_transform):
        """
        Exhaustive search over rotations only (translations fixed at the centered
        initialization) to seed the fine stage and avoid poor local minima.
        """
        reg = sitk.ImageRegistrationMethod()

        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=self.n_histogram_bins)
        reg.SetMetricSamplingStrategy(reg.REGULAR)
        reg.SetMetricSamplingPercentage(self.sampling_percentage, seed=self.seed)
        reg.SetInterpolator(sitk.sitkLinear)

        # Euler3D parameter order: (angleX, angleY, angleZ, tx, ty, tz).
        n = self.n_coarse_steps
        reg.SetOptimizerAsExhaustive([n, n, n, 0, 0, 0])
        reg.SetOptimizerScales([
            self.coarse_rotation_step, self.coarse_rotation_step, self.coarse_rotation_step,
            1.0, 1.0, 1.0,
        ])

        reg.SetInitialTransform(initial_transform, inPlace=True)
        reg.Execute(fixed, moving)
        return initial_transform

    def register(self, moving_zyx, moving_spacing_zyx, fixed_zyx=None, fixed_spacing_zyx=None):
        """
        Register a moving CT volume to the fixed reference and return the aligned
        volume in the reference frame, as a (z, y, x) array.

        If no fixed volume is passed, the reference at config.reference_path is
        loaded and cached on first use.
        """
        if fixed_zyx is None:
            if self._reference is None:
                self.load_reference()
            fixed_zyx, fixed_spacing_zyx = self._reference

        fixed = self._to_sitk(fixed_zyx, fixed_spacing_zyx)
        moving = self._to_sitk(moving_zyx, moving_spacing_zyx)

        initial_transform = sitk.Euler3DTransform(
            sitk.CenteredTransformInitializer(
                fixed, moving, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        )

        initial_transform = self._coarse_rotation_search(fixed, moving, initial_transform)

        reg = self._build_registration()
        reg.SetInitialTransform(initial_transform, inPlace=False)
        final_transform = reg.Execute(fixed, moving)

        aligned = sitk.Resample(
            moving, fixed, final_transform, sitk.sitkLinear,
            self.fill_value, moving.GetPixelID(),
        )

        return self._to_numpy(aligned)
