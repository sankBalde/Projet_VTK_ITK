"""
Auteur:
    Abdoulaye Baldé
    Majeur Image 2024
"""

import itk
import vtk
import numpy as np

import matplotlib.pyplot as plt

def load_nrrd_image(file_path):
    image = itk.imread(file_path)
    return image

def display_image_plt(image, im_number = 0, title="Image"):
    array = itk.GetArrayViewFromImage(image)
    print("array shape: ", array.shape)
    plt.figure()
    plt.title(title)
    plt.imshow(array[im_number])
    plt.show()


def register_images_translation(fixed_image_path, moving_image_path):
    PixelType = itk.ctype("float")

    fixedImage = itk.imread(fixed_image_path, PixelType)
    movingImage = itk.imread(moving_image_path, PixelType)

    Dimension = fixedImage.GetImageDimension()
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]

    TransformType = itk.TranslationTransform[itk.D, Dimension]
    initialTransform = TransformType.New()

    optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
        LearningRate=4,
        MinimumStepLength=0.001,
        RelaxationFactor=0.5,
        NumberOfIterations=200,
    )

    metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()

    registration = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType].New(
        FixedImage=fixedImage,
        MovingImage=movingImage,
        Metric=metric,
        Optimizer=optimizer,
        InitialTransform=initialTransform,
    )

    movingInitialTransform = TransformType.New()
    initialParameters = movingInitialTransform.GetParameters()
    initialParameters[0] = 0
    initialParameters[1] = 0
    movingInitialTransform.SetParameters(initialParameters)
    registration.SetMovingInitialTransform(movingInitialTransform)

    identityTransform = TransformType.New()
    identityTransform.SetIdentity()
    registration.SetFixedInitialTransform(identityTransform)

    registration.SetNumberOfLevels(1)
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SetShrinkFactorsPerLevel([1])

    registration.Update()

    transform = registration.GetTransform()
    finalParameters = transform.GetParameters()
    translationAlongX = finalParameters.GetElement(0)
    translationAlongY = finalParameters.GetElement(1)

    numberOfIterations = optimizer.GetCurrentIteration()

    bestValue = optimizer.GetValue()

    print("Result = ")
    print(" Translation X = " + str(translationAlongX))
    print(" Translation Y = " + str(translationAlongY))
    print(" Iterations    = " + str(numberOfIterations))
    print(" Metric value  = " + str(bestValue))

    CompositeTransformType = itk.CompositeTransform[itk.D, Dimension]
    outputCompositeTransform = CompositeTransformType.New()
    outputCompositeTransform.AddTransform(movingInitialTransform)
    outputCompositeTransform.AddTransform(registration.GetModifiableTransform())

    resampler = itk.ResampleImageFilter[MovingImageType, FixedImageType].New(
        Input=movingImage,
        Transform=outputCompositeTransform,
        UseReferenceImage=True,
        ReferenceImage=fixedImage,
    )
    resampler.SetDefaultPixelValue(100)

    resampler.Update()
    registered_image = resampler.GetOutput()

    return fixedImage, registered_image


def display_mutiliples_plt(images, titles=None, im_number=0):
    if titles is None:
        titles = ["Image"] * len(images)
    elif len(titles) != len(images):
        raise ValueError("The number of titles must match the number of images.")

    num_images = len(images)

    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))

    if num_images == 1:
        axs = [axs]

    for idx, (image, title) in enumerate(zip(images, titles)):
        array = itk.GetArrayViewFromImage(image)
        print(f"array shape for image {idx}: ", array.shape)
        axs[idx].imshow(array[im_number])
        axs[idx].set_title(title)
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()


def rigid2D(fixed_image, moving_image):

    if fixed_image.GetImageDimension() != 2 or moving_image.GetImageDimension() != 2:
        raise ValueError("Both images must be 2D.")

    dimension = fixed_image.GetImageDimension()
    FixedImageType = type(fixed_image)
    MovingImageType = type(moving_image)

    TransformType = itk.Rigid2DTransform[itk.D]
    initialTransform = TransformType.New()

    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()

    optimizer.SetLearningRate(4)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)

    metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()

    registration = itk.ImageRegistrationMethodv4.New(FixedImage=fixed_image, MovingImage=moving_image, Metric=metric,
                                                     Optimizer=optimizer, InitialTransform=initialTransform)

    moving_initial_transform = TransformType.New()
    initial_parameters = moving_initial_transform.GetParameters()
    initial_parameters[0] = 0
    initial_parameters[1] = 0
    initial_parameters[2] = 0
    moving_initial_transform.SetParameters(initial_parameters)


    scale_parameters = moving_initial_transform.GetParameters()
    scale_parameters[0] = 1000
    scale_parameters[1] = 1
    scale_parameters[2] = 1
    optimizer.SetScales(scale_parameters)

    registration.SetMovingInitialTransform(moving_initial_transform)


    fixed_parameters = moving_initial_transform.GetFixedParameters()
    fixed_parameters[0] = moving_image.GetLargestPossibleRegion().GetSize()[0] / 2.0
    fixed_parameters[1] = moving_image.GetLargestPossibleRegion().GetSize()[1] / 2.0

    moving_initial_transform.SetFixedParameters(fixed_parameters)

    identity_transform = TransformType.New()
    identity_transform.SetIdentity()
    registration.SetFixedInitialTransform(identity_transform)
    registration.SetNumberOfLevels(1)

    registration.Update()

    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()
    angle = final_parameters.GetElement(0)
    translation_along_x = final_parameters.GetElement(1)
    translation_along_y = final_parameters.GetElement(2)

    number_of_iterations = optimizer.GetCurrentIteration()

    best_value = optimizer.GetValue()

    print("Result = ")
    print(" Angle = " + str(angle))
    print(" Translation X = " + str(translation_along_x))
    print(" Translation Y = " + str(translation_along_y))
    print(" Iterations    = " + str(number_of_iterations))
    print(" Metric value  = " + str(best_value))

    CompositeTransformType = itk.CompositeTransform[itk.D, dimension]
    output_composite_transform = CompositeTransformType.New()
    output_composite_transform.AddTransform(moving_initial_transform)
    output_composite_transform.AddTransform(registration.GetModifiableTransform())

    resampler = itk.ResampleImageFilter.New(Input=moving_image, Transform=transform, UseReferenceImage=True,
                                            ReferenceImage=fixed_image)
    resampler.SetDefaultPixelValue(100)
    resampler.Update()
    registered_image = resampler.GetOutput()

    return registered_image

def rigid_info_mutuelle(fixed_image,
         moving_image):

    if fixed_image.GetImageDimension() != 2 or moving_image.GetImageDimension() != 2:
        raise ValueError("Both images must be 2D.")
    dimension = fixed_image.GetImageDimension()
    FixedImageType = type(fixed_image)
    MovingImageType = type(moving_image)

    TransformType = itk.Rigid2DTransform[itk.D]
    initialTransform = TransformType.New()

    optimizer = itk.RegularStepGradientDescentOptimizerv4.New()

    optimizer.SetLearningRate(4)
    optimizer.SetMinimumStepLength(0.001)
    optimizer.SetNumberOfIterations(200)

    metric = itk.MattesMutualInformationImageToImageMetricv4[FixedImageType, MovingImageType].New()

    registration = itk.ImageRegistrationMethodv4.New(FixedImage=fixed_image, MovingImage=moving_image, Metric=metric,
                                                     Optimizer=optimizer, InitialTransform=initialTransform, )

    moving_initial_transform = TransformType.New()
    initial_parameters = moving_initial_transform.GetParameters()
    initial_parameters[0] = 0
    initial_parameters[1] = 0
    initial_parameters[2] = 0
    moving_initial_transform.SetParameters(initial_parameters)


    scale_parameters = moving_initial_transform.GetParameters()
    scale_parameters[0] = 1000
    scale_parameters[1] = 1
    scale_parameters[2] = 1
    optimizer.SetScales(scale_parameters)

    registration.SetMovingInitialTransform(moving_initial_transform)


    fixed_parameters = moving_initial_transform.GetFixedParameters()
    fixed_parameters[0] = moving_image.GetLargestPossibleRegion().GetSize()[0] / 2.0
    fixed_parameters[1] = moving_image.GetLargestPossibleRegion().GetSize()[1] / 2.0

    moving_initial_transform.SetFixedParameters(fixed_parameters)

    identity_transform = TransformType.New()
    identity_transform.SetIdentity()
    registration.SetFixedInitialTransform(identity_transform)
    registration.SetNumberOfLevels(1)

    registration.Update()

    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()
    angle = final_parameters.GetElement(0)
    translation_along_x = final_parameters.GetElement(1)
    translation_along_y = final_parameters.GetElement(2)

    number_of_iterations = optimizer.GetCurrentIteration()

    best_value = optimizer.GetValue()

    print("Result = ")
    print(" Angle = " + str(angle))
    print(" Translation X = " + str(translation_along_x))
    print(" Translation Y = " + str(translation_along_y))
    print(" Iterations    = " + str(number_of_iterations))
    print(" Metric value  = " + str(best_value))

    CompositeTransformType = itk.CompositeTransform[itk.D, dimension]
    output_composite_transform = CompositeTransformType.New()
    output_composite_transform.AddTransform(moving_initial_transform)
    output_composite_transform.AddTransform(registration.GetModifiableTransform())

    resampler = itk.ResampleImageFilter.New(Input=moving_image, Transform=transform, UseReferenceImage=True,
                                            ReferenceImage=fixed_image, )
    resampler.SetDefaultPixelValue(100)
    resampler.Update()

    registered_image = resampler.GetOutput()

    return registered_image

def affine2D(fixed_image, moving_image):

    if fixed_image.GetImageDimension() != 2 or moving_image.GetImageDimension() != 2:
        raise ValueError("Both images must be 2D.")

    FixedImageType = type(fixed_image)
    MovingImageType = type(moving_image)

    TransformType = itk.AffineTransform[itk.D, 2]
    initialTransform = TransformType.New()

    optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
        LearningRate=4,
        MinimumStepLength=0.001,
        NumberOfIterations=200
    )

    metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()

    registration = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType].New(
        FixedImage=fixed_image,
        MovingImage=moving_image,
        Metric=metric,
        Optimizer=optimizer,
        InitialTransform=initialTransform
    )

    moving_initial_transform = TransformType.New()
    initial_parameters = moving_initial_transform.GetParameters()
    initial_parameters.Fill(0)
    moving_initial_transform.SetParameters(initial_parameters)

    scale_parameters = moving_initial_transform.GetParameters()
    scale_parameters.Fill(1)
    optimizer.SetScales(scale_parameters)

    registration.SetMovingInitialTransform(moving_initial_transform)

    fixed_parameters = moving_initial_transform.GetFixedParameters()
    fixed_parameters.Fill(fixed_image.GetLargestPossibleRegion().GetSize()[0] / 2.0)
    moving_initial_transform.SetFixedParameters(fixed_parameters)

    identity_transform = TransformType.New()
    identity_transform.SetIdentity()
    registration.SetFixedInitialTransform(identity_transform)
    registration.SetNumberOfLevels(1)

    registration.Update()

    transform = registration.GetTransform()
    final_parameters = transform.GetParameters()


    final_parameters_array = np.array(final_parameters)


    matrix = np.array(final_parameters_array[:6]).reshape(2, 3)
    translation = np.array(final_parameters_array[6:8])

    number_of_iterations = optimizer.GetCurrentIteration()
    best_value = optimizer.GetValue()

    print("Result = ")
    print(" Affine Matrix = ")
    print(matrix)
    print(" Translation = " + str(translation))
    print(" Iterations    = " + str(number_of_iterations))
    print(" Metric value  = " + str(best_value))

    CompositeTransformType = itk.CompositeTransform[itk.D, 2]
    output_composite_transform = CompositeTransformType.New()
    output_composite_transform.AddTransform(moving_initial_transform)
    output_composite_transform.AddTransform(registration.GetModifiableTransform())

    resampler = itk.ResampleImageFilter[MovingImageType, FixedImageType].New(
        Input=moving_image,
        Transform=output_composite_transform,
        UseReferenceImage=True,
        ReferenceImage=fixed_image
    )
    resampler.SetDefaultPixelValue(100)
    resampler.Update()
    registered_image = resampler.GetOutput()

    return registered_image

def register_images_type(fixed_image_path, moving_image_path, type="rigid"):
    PixelType = itk.ctype("float")

    fixedImage = itk.imread(fixed_image_path, PixelType)
    movingImage = itk.imread(moving_image_path, PixelType)


    if fixedImage.GetImageDimension() != 3 or movingImage.GetImageDimension() != 3:
        raise ValueError("Both images must be 3D.")


    fixed_slices = [itk.GetImageFromArray(itk.GetArrayViewFromImage(fixedImage)[i, :, :]) for i in range(fixedImage.GetLargestPossibleRegion().GetSize()[2])]
    moving_slices = [itk.GetImageFromArray(itk.GetArrayViewFromImage(movingImage)[i, :, :]) for i in range(movingImage.GetLargestPossibleRegion().GetSize()[2])]

    if len(fixed_slices) != len(moving_slices):
        raise ValueError("The number of slices in both images must match.")


    registered_slices = []
    for i in range(len(fixed_slices)):
        if type == "rigid":
            registered_slice = rigid2D(fixed_slices[i], moving_slices[i])
        elif type == "affine":
            registered_slice = affine2D(fixed_slices[i], moving_slices[i])
        elif type == "info_mutuelle":
            registered_slice = rigid_info_mutuelle(fixed_slices[i], moving_slices[i])
        else:
            raise ValueError("Unknown transformation type. Use 'rigid', 'affine'")
        registered_slices.append(itk.GetArrayViewFromImage(registered_slice))


    registered_image_array = np.stack(registered_slices, axis=0)
    registered_image_3d = itk.GetImageFromArray(registered_image_array)
    registered_image_3d.SetSpacing(fixedImage.GetSpacing())
    registered_image_3d.SetOrigin(fixedImage.GetOrigin())
    registered_image_3d.SetDirection(fixedImage.GetDirection())

    return fixedImage, registered_image_3d


def segment_tumor_2d(slice_2d, seedX=110, seedY=100, lower=180, upper=255):

    if not isinstance(slice_2d, itk.Image):
        raise ValueError("The slice must be of type itk.Image.")


    smoother = itk.GradientAnisotropicDiffusionImageFilter.New(Input=slice_2d, NumberOfIterations=20, TimeStep=0.04,
                                                               ConductanceParameter=3)
    smoother.Update()

    connected_threshold = itk.ConnectedThresholdImageFilter.New(Input=smoother.GetOutput())
    connected_threshold.SetReplaceValue(255)
    connected_threshold.SetLower(lower)
    connected_threshold.SetUpper(upper)


    seed = itk.Index[2]()
    seed.SetElement(0, seedX)
    seed.SetElement(1, seedY)
    connected_threshold.AddSeed(seed)

    connected_threshold.Update()

    in_type = type(connected_threshold.GetOutput())
    output_type = itk.Image[itk.UC, 2]
    rescaler = itk.RescaleIntensityImageFilter[in_type, output_type].New(Input=connected_threshold.GetOutput())
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
    rescaler.Update()

    return rescaler.GetOutput()


def segment_tumor_v2(input_image, seedX=110, seedY=100, lower=180, upper=255):
    dimension = input_image.GetImageDimension()
    if dimension != 3:
        raise ValueError("The input image must be 3D.")

    size = input_image.GetLargestPossibleRegion().GetSize()


    slices = [itk.GetImageFromArray(itk.GetArrayViewFromImage(input_image)[i, :, :]) for i in range(size[2])]


    segmented_slices = []
    for slice_2d in slices:
        segmented_slice = segment_tumor_2d(slice_2d, seedX, seedY, lower, upper)
        segmented_slices.append(itk.GetArrayViewFromImage(segmented_slice))


    segmented_image_array = np.stack(segmented_slices, axis=0)
    segmented_image_3d = itk.GetImageFromArray(segmented_image_array)
    segmented_image_3d.SetSpacing(input_image.GetSpacing())
    segmented_image_3d.SetOrigin(input_image.GetOrigin())
    segmented_image_3d.SetDirection(input_image.GetDirection())

    return segmented_image_3d

def calculate_volume(segmented_image):
    array = itk.GetArrayViewFromImage(segmented_image)
    volume = np.sum(array > 0)
    return volume

def calculate_intensity_difference(fixed_tumor, moving_tumor):
    fixed_array = itk.GetArrayViewFromImage(fixed_tumor)
    moving_array = itk.GetArrayViewFromImage(moving_tumor)

    difference = np.abs(fixed_array - moving_array)

    return difference

def display_slice(image, slice_index, title="Slice"):
    array = itk.GetArrayViewFromImage(image)[slice_index, :, :]
    plt.imshow(array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def segment_tumor(image, lower_threshold=50, upper_threshold=150):
    PixelType = itk.template(image)[1][0]
    Dimension = image.GetImageDimension()
    ImageType = itk.Image[PixelType, Dimension]

    threshold_filter = itk.BinaryThresholdImageFilter[ImageType, ImageType].New()
    threshold_filter.SetInput(image)
    threshold_filter.SetLowerThreshold(lower_threshold)
    threshold_filter.SetUpperThreshold(upper_threshold)
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)

    threshold_filter.Update()
    segmented_image = threshold_filter.GetOutput()

    return segmented_image

def visualize_slices(image):

    vtk_image = itk.vtk_image_from_image(image)

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)


    dims = vtk_image.GetDimensions()


    slider_rep = vtk.vtkSliderRepresentation2D()
    slider_rep.SetMinimumValue(0)
    slider_rep.SetMaximumValue(dims[2] - 1)
    slider_rep.SetValue(0)
    slider_rep.SetTitleText("Slice")
    slider_rep.GetSliderProperty().SetColor(1, 0, 0)
    slider_rep.GetTitleProperty().SetColor(1, 0, 0)
    slider_rep.GetLabelProperty().SetColor(1, 0, 0)
    slider_rep.GetSelectedProperty().SetColor(0, 1, 0)
    slider_rep.GetTubeProperty().SetColor(1, 1, 0)
    slider_rep.GetCapProperty().SetColor(1, 1, 0)
    slider_rep.SetSliderLength(0.02)
    slider_rep.SetSliderWidth(0.03)
    slider_rep.SetEndCapLength(0.01)
    slider_rep.SetEndCapWidth(0.03)
    slider_rep.SetTubeWidth(0.005)
    slider_rep.SetLabelFormat("%3.0lf")
    slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint1Coordinate().SetValue(0.1, 0.1)
    slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_rep.GetPoint2Coordinate().SetValue(0.9, 0.1)

    slider_widget = vtk.vtkSliderWidget()
    slider_widget.SetInteractor(render_window_interactor)
    slider_widget.SetRepresentation(slider_rep)
    slider_widget.SetAnimationModeToAnimate()

    image_viewer = vtk.vtkImageViewer2()
    image_viewer.SetInputData(vtk_image)
    image_viewer.SetSlice(0)
    image_viewer.SetRenderer(renderer)
    image_viewer.SetRenderWindow(render_window)


    renderer.GetActiveCamera().ParallelProjectionOn()
    image_center = [dims[0] // 2, dims[1] // 2, 0]
    renderer.GetActiveCamera().SetFocalPoint(image_center)
    renderer.GetActiveCamera().SetPosition(image_center[0], image_center[1], -dims[2])
    renderer.ResetCamera()

    def update_slice(obj, event):
        slider_value = int(slider_rep.GetValue())
        image_viewer.SetSlice(slider_value)
        render_window.Render()

    slider_widget.AddObserver("InteractionEvent", update_slice)


    slider_widget.EnabledOn()
    image_viewer.Render()
    render_window_interactor.Start()

def visualize_contours(fixed_tumor):
    def create_contour_actor(image, value, color):

        vtk_image = itk.vtk_image_from_image(image)


        contour_filter = vtk.vtkContourFilter()
        contour_filter.SetInputData(vtk_image)
        contour_filter.SetValue(0, value)
        contour_filter.Update()


        contour_mapper = vtk.vtkPolyDataMapper()
        contour_mapper.SetInputConnection(contour_filter.GetOutputPort())
        contour_mapper.ScalarVisibilityOff()


        contour_actor = vtk.vtkActor()
        contour_actor.SetMapper(contour_mapper)
        contour_actor.GetProperty().SetColor(color)

        return contour_actor


    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)


    fixed_actor = create_contour_actor(fixed_tumor, 255, (1.0, 0.0, 0.0))

    renderer.AddActor(fixed_actor)


    style = vtk.vtkInteractorStyleTrackballCamera()
    render_window_interactor.SetInteractorStyle(style)


    renderer.SetBackground(0.1, 0.2, 0.4)
    render_window.Render()
    render_window_interactor.Start()
