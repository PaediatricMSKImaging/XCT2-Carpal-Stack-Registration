import numpy as np
import SimpleITK as sitk


def extract_xy_slice_compat(img3d: sitk.Image, k: int) -> sitk.Image:
    """
    Extract a single XY slice from a 3D SimpleITK image.

    Parameters
    ----------
    img3d : sitk.Image
        Input 3D SimpleITK image from which the slice will be extracted.
        The image is assumed to be indexed as (x, y, z).
    k : int
        Index (0-based) of the slice along the z-axis to extract.

    Returns
    -------
    sitk.Image
        2D SimpleITK image corresponding to the XY slice at index `k`.
    """
    size = list(img3d.GetSize())
    index = [0, 0, int(k)]
    size[2] = 0

    # Extract the 2D slice from the 3D image
    f = sitk.ExtractImageFilter()
    f.SetSize(size)
    f.SetIndex(index)
    f.SetDirectionCollapseToSubmatrix()

    out = f.Execute(img3d)

    # Ensure non-zero spacing
    sp = out.GetSpacing()
    if sp[0] <= 0 or sp[1] <= 0:
        sx, sy, _ = img3d.GetSpacing()
        out.SetSpacing((float(sx), float(sy)))

    return out


def reg_2d_slice(fixed: sitk.Image, moving: sitk.Image, k: int) -> sitk.Euler3DTransform:
    """
    Register a 2D moving slice to a fixed slice and lift the result to 3D.

    This function performs a 2D rigid registration (Euler2D) of `moving` to
    `fixed` using a correlation metric, multi-resolution framework, and
    gradient descent optimizer. The final 2D transform is then converted 
    into a 3D Euler transform whose center lies on slice index `k`.

    Parameters
    ----------
    fixed : sitk.Image
        2D SimpleITK image used as the fixed/reference image in the registration.
    moving : sitk.Image
        2D SimpleITK image used as the moving image in the registration.
    k : int
        Index of the slice in the corresponding 3D volume.
        This index is used as the z-coordinate of the 3D transform center,
        ensuring that the 2D transform is embedded at the correct slice.

    Returns
    -------
    sitk.Euler3DTransform
        3D Euler transform corresponding to the estimated 2D in-plane
        rotation and translation at slice index `k`. The transform has:
        - Rotation only around the z-axis (angles around x and y are zero)
        - Translation only in x and y
        - Center at (center_x_2d, center_y_2d, k)
    """
    # Initialize the 2D Euler transform (rotation + translation) 
    initial_tx = sitk.Euler2DTransform()
    initial_tx = sitk.CenteredTransformInitializer(
        fixed, moving, initial_tx,
        sitk.CenteredTransformInitializerFilter.MOMENTS
    )

    # 2D registration method
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(1.0, 42)
    R.SetInterpolator(sitk.sitkBSpline)
    R.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-8,
            numberOfIterations=500,
            gradientMagnitudeTolerance=1e-12
        )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([2,2,1,1])
    R.SetSmoothingSigmasPerLevel([1.0, 0.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(initial_tx, inPlace=False)

    # Run registration
    final_tx = R.Execute(fixed, moving)

    # Extract 2D transform parameters: (angle, translation_x, translation_y)
    parameters_2d = final_tx.GetParameters()
    fixed_parameters_2d = final_tx.GetFixedParameters()
    
    angle_2d_rad = parameters_2d[0]
    translation_x_2d = parameters_2d[1]
    translation_y_2d = parameters_2d[2]

    # Center of rotation in the 2D plane (x, y)
    center_x_2d = fixed_parameters_2d[0]
    center_y_2d = fixed_parameters_2d[1]

    # 2D to 3D transformation
    angle_x_3d = 0.0
    angle_y_3d = 0.0
    angle_z_3d = angle_2d_rad

    euler_3d_transform = sitk.Euler3DTransform()

    euler_3d_transform.SetCenter([center_x_2d, center_y_2d, k])
    euler_3d_transform.SetRotation(angle_x_3d, angle_y_3d, angle_z_3d)
    euler_3d_transform.SetTranslation([translation_x_2d, translation_y_2d, 0.0])

    return euler_3d_transform
