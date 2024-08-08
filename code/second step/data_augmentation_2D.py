from torch.utils.data import Dataset
import numpy as np
from functools import lru_cache
from torchvision import transforms
import SimpleITK as sitk
from pathlib import Path
import imageio
import glob
import os
from os import path
import SimpleITK as sitk
import cupy
import gryds

def resample(image, transform, interpolator=sitk.sitkLinear, default_value=0.0):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image

    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

def basename(arg):
    try:
        return os.path.splitext(os.path.basename(arg))[0]
    except Exception as e:
        if isinstance(arg, list):
            return [basename(el) for el in arg]
        else:
            raise e


def saveImage(fname, arr, spacing=None, dtype=np.float32):
    if type(spacing) == type(None):
        spacing = np.ones((len(arr.shape),))
    img = sitk.GetImageFromArray(arr.astype(dtype))
    img.SetSpacing(spacing[::-1])
    sitk.WriteImage(img, fname, True)

def loadMHD(fname):
    img = sitk.ReadImage(str(fname)) #[z,y,x]
    spacing = img.GetSpacing()[::-1]
    offset = img.GetOrigin()[::-1]
    img = sitk.GetArrayFromImage(img)
    spacing = np.asarray(spacing)
    offset = np.asarray(offset)
    return img, spacing, offset

def log(x, logfn=np.log10):
    epsilon = 1
    mask = x < 0
    y = np.empty_like(x, np.float32)
    y[~mask] = logfn(x[~mask]+epsilon)
    y[mask] = -logfn(-1*(x[mask]-epsilon))
    return y

def exp(x, log_inv_fn):
    epsilon = 1
    mask = x < 0
    y = np.empty_like(x, np.float32)
    y[~mask] = log_inv_fn(x[~mask]) - epsilon
    y[mask] = -log_inv_fn(-1*x[mask]) + epsilon
    return y

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, spacing = sample['image'], sample['landmarks'], sample['spacing']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks, 'spacing': spacing}


class RandomCrop:
    def __init__(self, output_size, rs=np.random):
        self.rs = rs
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, spacing = sample['image'], sample['landmarks'], sample['spacing']

        h, w = image.shape
        new_h, new_w = self.output_size

        top = self.rs.randint(0, h - new_h)
        left = self.rs.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        warped_landmarks = landmarks - [left, top]
        1/0

        return {'image': image, 'landmarks': warped_landmarks, 'spacing': spacing}


class RandomAffine(object):
    def __init__(self, p=.5, rs=np.random):
        self.p = p
        self.rs = rs

    def __call__(self, sample):
        image, landmarks, spacing = sample['image'], sample['landmarks'], sample['spacing']
        warped = image
        warped_landmarks = landmarks
        if self.rs.rand() > self.p:
            pass  # do nothing
        else:
            img = sitk.GetImageFromArray(image)
            new_transform = sitk.AffineTransform(2)
            matrix = np.identity(2, float)
            matrix += np.random.uniform(-.2, .2, matrix.shape)
            new_transform.SetMatrix(matrix.ravel())
            wrp = resample(img, new_transform)

            warped_landmarks = landmarks @ np.linalg.inv(matrix).T  # (np.linalg.inv(matrix)@landmarks.T).T

            warped = sitk.GetArrayFromImage(wrp)

        return {'image': warped,
                'landmarks': warped_landmarks,
                'spacing': spacing}

class PadTodivisable:
    def __init__(self, downsample_factor):
        self.downsample_factor = downsample_factor
    def __call__(self, sample):
        image, landmarks, spacing = sample['image'], sample['landmarks'], sample['spacing']
        padding = (self.downsample_factor - np.array(image.shape) % self.downsample_factor)
        padding = [(0, p) for p in padding]
        padded_image = np.pad(image, padding, mode="edge")
        return {'image': padded_image, 'landmarks': landmarks, 'spacing': spacing}



class RandomCrop3D:
    def __init__(self, output_size, rs=np.random):
        self.rs = rs
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, spacing = sample['image'], sample['landmarks'], sample['spacing']

        new_d, new_h, new_w = self.output_size
        # if output_size is bigger than image, padding is necessary
        #path_z = max(0, new_h - image.shape[0] + 1)
        #path_y = max(0, new_w - image.shape[1] + 1)
        #path_x = max(0, new_l - image.shape[2] + 1)
        #image = np.pad(image, ((0, path_z), (0, path_y), (0, path_x)), mode="constant")

        d, h, w = image.shape

        zstart = self.rs.randint(0, d - new_d)
        ystart = self.rs.randint(0, h - new_h)
        xstart = self.rs.randint(0, w - new_w)



        cropped_image = image[zstart: zstart + new_d,
                              ystart: ystart + new_h,
                              xstart: xstart + new_w]

        warped_landmarks = landmarks - np.array([[xstart, ystart, zstart]])

        # warped_landmarks = landmarks - np.array([[zstart, ystart, xstart]])

        return {'image': cropped_image, 'landmarks': warped_landmarks, 'spacing': spacing}


class RandomAffine3D(object):
    '''
    Expects landmarks to be annotated in z,y,x order.
    But transform is in x,y,z.
    '''
    def __init__(self, p=.5, rs=np.random, bg_value=0.0):
        self.p = p
        self.rs = rs
        self.bg_value = bg_value

    def __call__(self, sample):
        rand = self.rs.rand()
        image, landmarks, spacing = sample['image'], sample['landmarks'], sample['spacing']
        warped = image
        warped_landmarks = landmarks

        if rand > self.p:
            pass  # do nothing
        else:
            img = sitk.GetImageFromArray(image)
            origin = -np.array(image.shape[::-1], dtype=float) / 2
            img.SetOrigin(origin)
            new_transform = sitk.AffineTransform(3)
            matrix = np.identity(3, float)
            matrix += self.rs.uniform(-.2, .2, matrix.shape)
            new_transform.SetMatrix(matrix.ravel())
            wrp = resample(img, new_transform, interpolator=sitk.sitkLinear, default_value=self.bg_value)
            warped_landmarks = landmarks + origin
            warped_landmarks = (np.linalg.inv(matrix) @ warped_landmarks.T).T
            warped_landmarks = warped_landmarks - origin

            #warped_landmarks = landmarks + origin[::-1]
            #warped_landmarks = (np.linalg.inv(matrix) @ warped_landmarks[:, ::-1].T).T[:, ::-1]
            #warped_landmarks = warped_landmarks - origin[::-1]

            warped = sitk.GetArrayFromImage(wrp)

        return {'image': warped,
                'landmarks': warped_landmarks,
                'spacing': spacing}


def find_maximum_coord_in_image(image):
    # calculate maximum
    max_index = np.argmax(image)
    coord = np.array(np.unravel_index(max_index, image.shape), np.int32)
    return coord


def find_maximum_in_image(image):
    # calculate maximum
    coord = find_maximum_coord_in_image(image)
    max_value = image[tuple(coord)]
    # flip indizes from [y,x] to [x,y]
    return max_value, coord


def find_quadratic_subpixel_maximum_in_image(image):
    coord = find_maximum_coord_in_image(image)
    max_value = image[tuple(coord)]
    refined_coord = coord.astype(np.float32)
    dim = coord.size
    for i in range(dim):
        if int(coord[i]) - 1 < 0 or int(coord[i]) + 1 >= image.shape[i]:
            continue
        before_coord = coord.copy()
        before_coord[i] -= 1
        after_coord = coord.copy()
        after_coord[i] += 1
        pa = image[tuple(before_coord)]
        pb = image[tuple(coord)]
        pc = image[tuple(after_coord)]
        diff = 0.5 * (pa - pc) / (pa - 2 * pb + pc)
        refined_coord[i] += diff
    return max_value, refined_coord


def transform_landmarks_inverse(landmarks, transformation, size, spacing):
    """
    from Payer et al.
    Transforms a landmark object with the inverse of a given sitk transformation. If the transformation
    is not invertible, calculates the inverse by resampling from a dispacement field.
    :param landmarks: The landmark objects.
    :param transformation: The sitk transformation.
    :param size: The size of the output image, on which the landmark should exist.
    :param spacing: The spacing of the output image, on which the landmark should exist.
    :return: The landmark object with transformed coords.
    """
    try:
        inverse = transformation.GetInverse()
        transformed_landmarks = transform_landmarks(landmarks, inverse)
        for transformed_landmark in transformed_landmarks:
            if transformed_landmark.is_valid:
                transformed_landmark.coords /= np.array(spacing)
        return transformed_landmarks
    except:
        # consider a distance of 2 pixels as a maximum allowed distance
        # for calculating the inverse with a transformation field
        max_min_distance = np.max(spacing) * 2
        return transform_landmarks_inverse_with_resampling(landmarks, transformation, size, spacing, max_min_distance)


def transform_landmarks_inverse_with_resampling(landmarks, transformation, size, spacing, max_min_distance=None):
    """
    from Payer et al.
    Transforms a list of landmarks by calculating the inverse of a given sitk transformation by resampling from a displacement field.
    :param landmarks: The list of landmark objects.
    :param transformation: The sitk transformation.
    :param size: The size of the output image, on which the landmark should exist.
    :param spacing: The spacing of the output image, on which the landmark should exist.
    :param max_min_distance: The maximum distance of the coordinate calculated by resampling. If the calculated distance is larger than this value, the landmark will be set to being invalid.
    :return: The landmark object with transformed coords.
    """
    transformed_landmarks = landmarks.copy()
    dim = len(size)
    assert (dim == 3)
    displacement_field = sitk.TransformToDisplacementField(transformation,
                                                           sitk.sitkVectorFloat32,
                                                           size=size,
                                                           outputSpacing=spacing)
    displacement_field = sitk.GetArrayFromImage(displacement_field).transpose([2, 1, 0, 3])

    mesh = np.meshgrid(np.array(range(size[0]), np.float32),
                       np.array(range(size[1]), np.float32),
                       np.array(range(size[2]), np.float32),
                       indexing='ij')
    displacement_field += (np.stack(mesh, axis=3) * np.array(spacing, np.float32).reshape((1, 1, 1, 3)))

    for idx in range(len(transformed_landmarks)):
        #            if (not transformed_landmarks[i].is_valid) or (transformed_landmarks[i].coords is None):
        #                continue
        lm = transformed_landmarks[idx]
        # calculate distances to current landmark coordinates
        vec = displacement_field - lm
        # distances = np.sqrt(vec[:, :, :, 0] ** 2 + vec[:, :, :, 1] ** 2 + vec[:, :, :, 2] ** 2)

        distances = np.linalg.norm(vec, axis=3)
        invert_min_distance, lm = find_quadratic_subpixel_maximum_in_image(-distances)

        lm = lm * spacing
        # lm = np.unravel_index(np.argmin(distances), distances.shape)

        # print(lm)

        min_distance = -invert_min_distance
        if max_min_distance is not None and min_distance > max_min_distance:
            raise RuntimeError('error in min_distance')
        #    transformed_landmarks[i].is_valid = False
        #    transformed_landmarks[i].coords = None
        else:
            transformed_landmarks[idx] = lm
    return transformed_landmarks



class RandomBSplineTransform:
    def __init__(self, max_deformation=30, grid_control_points=(5, 5, 5), bspline_order=3, ndim=3, bg_value=0.0,
                 resample_interpolator=sitk.sitkBSpline, chance_of_application=1.,
                 rs=np.random):
        self.max_deformation = max_deformation
        self.grid_control_points = grid_control_points
        self.bspline_order = bspline_order
        self.ndim = ndim
        self.mesh_size = [gcp - bspline_order for gcp in grid_control_points]
        self.bg_value = bg_value
        self.rs = rs
        self.resample_interpolator = resample_interpolator
        self.chance_of_application = chance_of_application

    def __call__(self, smp):
        if self.chance_of_application < 1.:
            if self.rs.rand() > self.chance_of_application:
                return smp

        origin = np.zeros(self.ndim)
        direction = np.eye(self.ndim).flatten()
        landmarks = smp['landmarks']
        image = sitk.GetImageFromArray(smp['image'])
        image.SetSpacing(smp['spacing'][::-1])
        image.SetOrigin(origin)
        size = image.GetSize()
        spacing = image.GetSpacing()
        physical_dimensions = np.multiply(size, spacing)

        transform = sitk.BSplineTransform(self.ndim, self.bspline_order)
        transform.SetTransformDomainOrigin(origin)
        transform.SetTransformDomainMeshSize(self.mesh_size)
        transform.SetTransformDomainPhysicalDimensions(physical_dimensions)
        transform.SetTransformDomainDirection(direction)

        transformation_parameters = (self.rs.rand(*self.grid_control_points, self.ndim) - .5) * (2 * self.max_deformation)
        transform.SetParameters(transformation_parameters.ravel())

        warped_image = sitk.GetArrayFromImage(resample(image, transform, self.resample_interpolator, default_value=self.bg_value))
        warped_landmarks = transform_landmarks_inverse_with_resampling(landmarks, transform, size, spacing)
        return {'image': warped_image, 'landmarks': warped_landmarks, 'spacing': smp['spacing']}


def transform_landmarks(landmarks, transform, size, spacing, max_min_distance=None):
    transformed_landmarks = landmarks.copy()

    mesh = np.meshgrid(np.array(range(size[0]), np.float32),
                       np.array(range(size[1]), np.float32),
                       np.array(range(size[2]), np.float32),
                       indexing='ij')
    #     mesh = np.meshgrid(np.linspace(0, 1, size[0], endpoint=False, dtype=np.float32),
    #                        np.linspace(0, 1, size[1], endpoint=False, dtype=np.float32),
    #                        np.linspace(0, 1, size[2], endpoint=False, dtype=np.float32),
    #                        indexing='ij')

    identity_grid = np.stack(mesh, axis=0) * np.array(spacing, np.float32).reshape((3, 1, 1, 1))
    identity_grid = identity_grid  # [::-1] # x, y, z order

    scale = tuple(np.array(size) - 1)
    transformed_field = transform(identity_grid.reshape((3, -1)), scale=scale).reshape(identity_grid.shape)

    for idx in range(len(transformed_landmarks)):
        lm = transformed_landmarks[idx]
        # calculate distances to current landmark coordinates
        vec = transformed_field - lm.reshape((3, 1, 1, 1))[::-1]

        distances = np.linalg.norm(vec, axis=0)
        invert_min_distance, lm = find_quadratic_subpixel_maximum_in_image(-distances)

        # lm = lm * spacing
        lm = lm[::-1] * spacing  # TODO: check if spacing works

        min_distance = -invert_min_distance
        if max_min_distance is not None and min_distance > max_min_distance:
            raise RuntimeError('error in min_distance')
        #    transformed_landmarks[i].is_valid = False
        #    transformed_landmarks[i].coords = None
        else:
            transformed_landmarks[idx] = lm
    return transformed_landmarks


class RandomBSplineTransformGryds:
    def __init__(self, deformation_amount=.1, grid_control_points=(5, 5, 5),
                 chance_of_application=1., rs=np.random, use_cuda=None):
        self.chance_of_application = chance_of_application
        self.grid_control_points = grid_control_points
        self.rs = rs
        self.cuda_device = 0
        self.deformation_amount = deformation_amount

        #        if use_cuda:
        #             self.
        self.Transformer = gryds.BSplineTransformationCuda
        self.Interpolator = gryds.BSplineInterpolatorCuda
        #self.Transformer = gryds.BSplineTransformation
        #self.Interpolator = gryds.BSplineInterpolator
        # TODO: check if spacing works

    def __call__(self, smp):
        if self.chance_of_application < 1.:
            if self.rs.rand() > self.chance_of_application:
                return smp

        with cupy.cuda.Device(self.cuda_device):
            image = smp['image']
            spacing = smp['spacing']
            landmarks = smp['landmarks']

            random_grid = self.rs.rand(image.ndim, 5, 5, 5)
            random_grid -= 0.5
            random_grid *= self.deformation_amount

            # bspline = gryds.BSplineTransformation(random_grid)
            bspline = self.Transformer(random_grid)

            # interpolator = gryds.Interpolator(image)
            interpolator = self.Interpolator(image)

            warped_image = interpolator.transform(bspline, mode='nearest')

            warped_landmarks = transform_landmarks(landmarks, bspline, image.shape, spacing)
            # print(landmarks)
            # print(warped_landmarks)
            return {'image': warped_image, 'landmarks': warped_landmarks, 'spacing': smp['spacing']}


class RandomIntensity(object):
    def __init__(self, rs=np.random):
        self.rs = rs
        # self.maximum_g = 1.25
        # self.maximum_gain = 10

    def __call__(self, sample):
        image, landmarks, spacing = sample['image'], sample['landmarks'], sample['spacing']

        # transform = self.rs.randint(2)
        # if transform == 0:
        #     pass
        # elif transform == 1:
        gain = self.rs.uniform(2.5, 7.5)
        cutoff = self.rs.uniform(0.25, 0.75)
        image = (1 / (1 + np.exp(gain * (cutoff - image))))

        # else:
        #     g = self.rs.rand() * 2 * self.maximum_g - self.maximum_g
        #     if g < 0:
        #         g = 1 / np.abs(g)
        #     image = image**g

        return {'image': image,
                'landmarks': landmarks,
                'spacing': spacing}


class RandomSubImagesAroundLandmarks:
    def __init__(self, size=16, rs=np.random):
        self.size = size
        self.rs = rs
        self.offset = size / 2

    def __call__(self, smp):
        image = smp['image']
        landmarks = smp['landmarks']
        spacing = smp['spacing']
        size = self.size

        random_start_idcs = landmarks.astype(int)[:, ::-1] - self.rs.randint(0, size, landmarks.shape)

        check_if_too_large = np.array(image.shape)[None] - (random_start_idcs + size)
        mask = check_if_too_large < 0
        random_start_idcs[mask] += check_if_too_large[mask]
        random_start_idcs[random_start_idcs<0] = 0
        # print(image.shape, random_start_idcs+size)
        sub_images = [image[si[0]:si[0] + size, si[1]:si[1] + size] for si in random_start_idcs]
        sub_images = np.stack(sub_images)
        # print(sub_images.shape)
        sub_landmarks = landmarks - random_start_idcs[:, ::-1]
        displacement = sub_landmarks - self.offset

        return {'image': sub_images, 'landmarks': sub_landmarks,
                'displacement': displacement, 'spacing': smp['spacing']}

class TestSubImagesAroundLandmarks:
    def __init__(self, size=16):
        self.size = size
        self.offset = size / 2

    def __call__(self, smp):
        image = smp['image']
        landmarks = smp['landmarks']
        estimated_landmarks = smp['estimated_landmarks']
        spacing = smp['spacing']
        size = self.size

        # random_start_idcs = landmarks.astype(int)[:, ::-1] - (size//2)
        random_start_idcs = estimated_landmarks.astype(int)[:, ::-1] - (size//2)


        check_if_too_large = np.array(image.shape)[None] - (random_start_idcs + size)
        mask = check_if_too_large < 0
        random_start_idcs[mask] += check_if_too_large[mask]
        random_start_idcs[random_start_idcs<0] = 0
        # print(image.shape, random_start_idcs+size)
        sub_images = [image[si[0]:si[0] + size, si[1]:si[1] + size] for si in random_start_idcs]
        sub_images = np.stack(sub_images)
        # print(sub_images.shape)
        sub_landmarks = landmarks - random_start_idcs[:, ::-1]
        displacement = sub_landmarks - self.offset

        return {'image': sub_images, 'landmarks': sub_landmarks,
                'displacement': displacement, 'spacing': smp['spacing'], 'name': smp['name']}
# plt.hist(transformation_parameters.ravel(), 100);

def create_landmark_classmask(shape, spacing, landmarks):
    class_mask = np.zeros([len(landmarks)] + list(shape), bool)
    idcs = (landmarks / spacing).astype(int)
    valid_landmarks = np.logical_and((idcs >= 0).all(1), np.less(idcs, shape).all(1))
    idcs_raveled = np.ravel_multi_index(idcs.T, dims=class_mask.shape[1:], mode='clip')
    idcs_raveled = idcs_raveled + np.cumsum([0]+[np.product(class_mask.shape[1:])]*(len(landmarks)-1))
    idcs_raveled = idcs_raveled[valid_landmarks]
    class_mask.flat[idcs_raveled] = True
    return class_mask


def create_identity_grid2D(shape, spacing, offset_center=False):
    spacing = np.array(spacing, float)
    shape = np.array(shape, float)
    extent = spacing * shape
    linear_coordinates = [np.linspace(0, ext, shp, endpoint=False) for ext, shp in zip(extent, shape)]
    grid = np.stack(np.meshgrid(*linear_coordinates, indexing='ij'))
    if offset_center:
        center_offset = spacing / 2
        grid = grid + center_offset[:, None, None]
    return grid

def create_landmark_grid2D(shape, spacing, landmarks):
    grid = create_identity_grid2D(shape, spacing, offset_center=True)
    grid = landmarks[:, :, None, None] - grid[None]
    return grid

class ConvertToTrainingSample2D(object):
    def __init__(self, downsample_factor, logfn=None, lm=-1):
        self.downsample_factor = downsample_factor
        self.logfn=logfn
        self.lm=lm

    def __call__(self, sample):
        images, landmarks = sample['image'], sample['landmarks']
        nr_ims = np.arange(images.shape[0])

        if self.lm >= 0:
            landmarks = landmarks[[self.lm]]
        shape = np.array(images.shape[1:]) // self.downsample_factor
        spacing = np.array(sample['spacing']) * self.downsample_factor

        cls = [create_landmark_classmask(shape, spacing, landmarks[x, None]).astype(np.float32) for x in nr_ims]
        cls = np.stack(cls)
        cls = np.squeeze(cls)
        rgr = [create_landmark_grid2D(shape, spacing, landmarks[x, None]).astype(np.float32) for x in nr_ims]
        rgr = np.stack(rgr)
        rgr = np.squeeze(rgr)

        if self.logfn != None:
            rgr = log(rgr, self.logfn)
        return {'image': images,
                'classes': cls,
                'displacement': rgr,
                'landmarks': landmarks}


def prediction_to_landmark(class_map, displacement_map, downsample_factor, spacing, log_inv_fn, accumulate_log=False):
    assert (class_map.ndim == 4)
    assert (displacement_map.ndim == 5)
    spacing = np.array(spacing) * downsample_factor

    grid = create_identity_grid3D(class_map.shape[1:], spacing, True)

    if accumulate_log:
        weighted_displacements = (displacement_map * class_map[:, np.newaxis])
        avg_displacement = weighted_displacements.sum((2, 3, 4)) / class_map.sum((1, 2, 3))[:, np.newaxis]
        avg_absolute_position = (grid[np.newaxis] * class_map[:, np.newaxis]).sum((2,3,4))
        landmarks = avg_absolute_position + exp(avg_displacement, log_inv_fn)
    else:
        absolute_positions = grid[np.newaxis] + exp(displacement_map, log_inv_fn)
        weighted_positions = absolute_positions * class_map[:, np.newaxis]
        landmarks = weighted_positions.sum((2, 3, 4)) / class_map.sum((1, 2, 3))[:, np.newaxis]
    # print((grid + exp(displacement_map, base) * class_map[:, np.newaxis]).mean((2,3,4)).shape)
    # landmarks = (grid + displacement_map * class_map[:, np.newaxis]).mean((2,3,4))
    return landmarks


class CephalometricDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset='train', root_dir='/home/julia/landmark_detection/paper/data/images/Cephalometric',
                 transform=None, single=False, lm_idx=None,):
        self.dataset = dataset
        self.root_dir = Path(root_dir + "/" + dataset + "/" + 'images_png')
        self.lab_dir = r"/home/julia/landmark_detection/paper/data/labels/Cephalometric/origineelAverage"
        self.transform = transform
        self.single = single
        self.lm_idx = lm_idx

        self.rs = np.random.RandomState(123)
        patids = list(Path(self.root_dir).glob('*.png'))
        self.patids = patids
        self.estimates = True
        if 'train' in dataset:
            self.estimates = False

    def __len__(self):
        return len(self.patids)

    def __getitem__(self, idx):
        if self.estimates:
            image, landmarks, estimated_lms = self.load_sample(self.patids[idx], self.estimates)
            sample = {'image': image, 'landmarks': landmarks, 'spacing': (1, 1), 'estimated_landmarks': estimated_lms, 'name': basename(self.patids[idx])}
        else:
            image, landmarks = self.load_sample(self.patids[idx], self.estimates)
            sample = {'image': image, 'landmarks': landmarks, 'spacing': (1, 1), 'name': basename(self.patids[idx])}

        if self.transform:
            sample = self.transform(sample)
        return sample


    @lru_cache()
    def load_sample(self, patid, estimates):
        image_fname = patid
        lab_fname = self.lab_dir + r"/" + basename(patid) + ".txt"
        estimated_fname = self.lab_dir + r'_EstimatedML/' + basename(patid) + ".txt"
        image = np.array(imageio.imread(image_fname))
        lms = np.loadtxt(lab_fname)  # [:-4]
        lms = np.asarray(lms)
        if self.single:
            lms = lms[self.lm_idx]
            lms = lms[None, :]
        if estimates:
            estimated_lms = np.loadtxt(estimated_fname, max_rows=19)
            estimated_lms = np.copy(estimated_lms[:, ::-1])  # from z,y,x to x,y,z order
            return image, lms, estimated_lms
        return image, lms

def rescale_intensities(im, dtype=np.float32):
    min_val, max_val = np.percentile(im, (1, 99))
    im = ((im.astype(dtype) - min_val) / (max_val - min_val)).clip(0, 1)
    return im


class CCTADataset(Dataset):
    """CCTA Landmarks dataset."""

    def __init__(self, dataset='train', root_dir='/home/julia/landmark_detection/paper/data/images/CCTA',
                 transform=None,
                 vs=1.5, single=False, lm_idx=None, select_first_three=False):
        if vs == 0.5:
            self.voxel_size = "/images_VS05/"
            self.lab_dir = r"/home/julia/landmark_detection/paper/data/labels/CCTA/VS_05"
        elif vs == 1.5:
            self.voxel_size = "/images_VS15/"
            self.lab_dir = r"/home/julia/landmark_detection/paper/data/labels/CCTA/VS_15"
        self.dataset = dataset
        self.root_dir = Path(root_dir + "/" + dataset + self.voxel_size)
        self.transform = transform
        self.single = single
        self.lm_idx = lm_idx

        patids = list(Path(self.root_dir).glob('*.nii.gz'))

        self.rs = np.random.RandomState(123)

        if select_first_three:
            patids = patids[:3]
        self.patids = patids
        self.estimates=True
        if 'train' in dataset:
            self.estimates=False


    def __len__(self):
        return len(self.patids)

    def __getitem__(self, idx):
        if self.estimates:
            image, landmarks, estimated_lms = self.load_sample(self.patids[idx], self.estimates)
            sample = {'image': image, 'landmarks': landmarks, 'spacing': (1, 1, 1), 'estimated_landmarks': estimated_lms, 'name': basename(self.patids[idx])}
        else:
            image, landmarks = self.load_sample(self.patids[idx], self.estimates)
            sample = {'image': image, 'landmarks': landmarks, 'spacing': (1, 1, 1), 'name': basename(self.patids[idx])}

        if self.transform:
            sample = self.transform(sample)
        return sample

    @lru_cache()
    def load_sample(self, patid, estimates):
        image_fname = patid
        lab_fname = self.lab_dir + r"/" + Path(patid).stem[:-4] + ".txt"
        estimated_fname = self.lab_dir + r'_EstimatedML/' + Path(patid).stem[:-4] + ".nii.txt"
        image, _, _ = loadMHD(image_fname)
        image = ((image.astype(np.float32) + 1000) / 4095).clip(0, 1)
        lms = np.loadtxt(lab_fname, max_rows=8)
        lms = np.copy(lms[:, ::-1])  # from z,y,x to x,y,z order
        if self.single:
            lms = lms[self.lm_idx]
            lms = lms[None, :]
        if estimates:
            estimated_lms = np.loadtxt(estimated_fname, max_rows=8)
            estimated_lms = np.copy(estimated_lms[:, ::-1])  # from z,y,x to x,y,z order
            return image, lms, estimated_lms
        return image, lms


class BulbusDataset(Dataset):
    """Bulbus Landmarks dataset."""

    def __init__(self, dataset='train', root_dir='/home/julia/landmark_detection/paper/data/images/Bulbus', transform=None,
                 chunk_size=None, single = False, lm_idx=None):
        self.voxel_size = "/images_VS047/"
        self.lab_dir = r"/home/julia/landmark_detection/paper/data/labels/Bulbus/isotropic_lm"

        self.dataset = dataset
        self.root_dir = Path(root_dir+"/"+ dataset + self.voxel_size)
        self.transform = transform

        patids = glob.glob(path.join(self.root_dir, '*.nii.gz'))

        self.rs = np.random.RandomState(123)
        self.patids = patids
        self.estimates = True
        if 'train' in dataset:
            self.estimates = False
        self.chunk_size=chunk_size
        self.single = single
        self.lm_idx = lm_idx

    def __len__(self):
        return len(self.patids)

    def __getitem__(self, idx):
        if self.estimates:
            image, landmarks, estimated_lms = self.load_sample(self.patids[idx], self.estimates)
            sample = {'image': image, 'landmarks': landmarks, 'spacing': (1, 1, 1),
                      'estimated_landmarks': estimated_lms, 'name': basename(self.patids[idx])}
        else:
            image, landmarks = self.load_sample(self.patids[idx], self.estimates)
            sample = {'image': image, 'landmarks': landmarks, 'spacing': (1, 1, 1), 'name': basename(self.patids[idx])}

        if self.transform:
                sample = self.transform(sample)

        return sample

    @lru_cache()
    def load_sample(self, patid, estimates):
        image_fname = patid
        lab_fname_right = self.lab_dir + r"/" + Path(patid).stem[:-4] + "_Rechts.txt"
        lab_fname_left = self.lab_dir +  r"/" + Path(patid).stem[:-4] + "_Links.txt"
        estimated_fname = self.lab_dir + r'_EstimatedML/' + Path(patid).stem[:-4] + ".nii.txt"

        image, _, _ = loadMHD(image_fname)
        z_pad = self.chunk_size
        if z_pad > 0:
            image = np.pad(image, ((0, z_pad + 1), (0, z_pad + 1), (0, z_pad + 1)), mode="constant")
        lm_right = np.loadtxt(lab_fname_right)
        lm_left = np.loadtxt(lab_fname_left)
        lms = [lm_left, lm_right]
        lms=np.asarray(lms)
        lms = np.copy(lms[:, ::-1])
        if self.single:
            lms = lms[self.lm_idx]
            lms = lms[None, :]
        if estimates:
            estimated_lms = np.loadtxt(estimated_fname, max_rows=2)
            estimated_lms = np.copy(estimated_lms[:, ::-1])  # from z,y,x to x,y,z order
            return image, lms, estimated_lms
        return image, lms