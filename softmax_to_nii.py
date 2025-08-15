
import numpy as np
import SimpleITK as sitk
import pickle
from typing import Union

def save_softmax_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray], out_fname: str,
                                         # properties_dict: dict, order: int = 1,
                                         # region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         # resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, # force_separate_z: bool = None,
                                         # interpolation_order_z: int = 0, verbose: bool = True
                                    ):
    """
    This is a utility for writing segmentations to nifto and npz. It requires the data to have been preprocessed by
    GenericPreprocessor because it depends on the property dictionary output (dct) to know the geometry of the original
    data. segmentation_softmax does not have to have the same size in pixels as the original data, it will be
    resampled to match that. This is generally useful because the spacings our networks operate on are most of the time
    not the native spacings of the image data.
    If seg_postprogess_fn is not None then seg_postprogess_fnseg_postprogess_fn(segmentation, *seg_postprocess_args)
    will be called before nifto export
    There is a problem with python process communication that prevents us from communicating obejcts
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code.) We circumvent that problem here by saving softmax_pred to a npy file that will
    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
    filename or np.ndarray for segmentation_softmax and will handle this automatically
    :param segmentation_softmax:
    :param out_fname:
    :param properties_dict:
    :param order:
    :param region_class_order:
    :param seg_postprogess_fn:
    :param seg_postprocess_args:
    :param resampled_npz_fname:
    :param non_postprocessed_fname:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately. Do not touch unless you know what you are doing
    :param interpolation_order_z: if separate z resampling is done then this is the order for resampling in z
    :param verbose:
    :return:
    """
    # if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)
    #
    # if isinstance(segmentation_softmax, str):
    #     assert isfile(segmentation_softmax), "If isinstance(segmentation_softmax, str) then " \
    #                                          "isfile(segmentation_softmax) must be True"
    #     del_file = deepcopy(segmentation_softmax)
    #     segmentation_softmax = np.load(segmentation_softmax)
    #     os.remove(del_file)
    #
    # # first resample, then put result into bbox of cropping, then save
    # current_shape = segmentation_softmax.shape
    # shape_original_after_cropping = properties_dict.get('size_after_cropping')
    # shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    # # current_spacing = dct.get('spacing_after_resampling')
    # # original_spacing = dct.get('original_spacing')
    #
    # if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
    #     if force_separate_z is None:
    #         if get_do_separate_z(properties_dict.get('original_spacing')):
    #             do_separate_z = True
    #             lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
    #         elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
    #             do_separate_z = True
    #             lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
    #         else:
    #             do_separate_z = False
    #             lowres_axis = None
    #     else:
    #         do_separate_z = force_separate_z
    #         if do_separate_z:
    #             lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
    #         else:
    #             lowres_axis = None
    #
    #     if lowres_axis is not None and len(lowres_axis) != 1:
    #         # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
    #         # separately in the out of plane axis
    #         do_separate_z = False
    #
    #     if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
    #     seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
    #                                            axis=lowres_axis, order=order, do_separate_z=do_separate_z, cval=0,
    #                                            order_z=interpolation_order_z)
    #     # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    # else:
    #     if verbose: print("no resampling necessary")
    #     seg_old_spacing = segmentation_softmax

    # if resampled_npz_fname is not None:
    #     np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
    #     # this is needed for ensembling if the nonlinearity is sigmoid
    #     if region_class_order is not None:
    #         properties_dict['regions_class_order'] = region_class_order
    #     save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")

    seg_old_spacing = np.load(segmentation_softmax)
    seg_old_spacing = seg_old_spacing.f.softmax

    file_to_read = open(segmentation_softmax[:-4] + ".pkl", "rb")
    properties_dict = pickle.load(file_to_read)

    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')

    # if region_class_order is None:
    #     seg_old_spacing = seg_old_spacing.argmax(0)
    # else:
    #     seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
    #     for i, c in enumerate(region_class_order):
    #         seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
    #     seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')
    bbox.insert(0, [0, seg_old_spacing.shape[0] + 1])
    print(type(shape_original_before_cropping))
    print(shape_original_before_cropping)
    shape_original_before_cropping = list(shape_original_before_cropping)
    shape_original_before_cropping.insert(0, seg_old_spacing.shape[0])
    shape_original_before_cropping = np.array(shape_original_before_cropping)
    print(seg_old_spacing.shape)
    print(shape_original_before_cropping)
    print(bbox)
    # return bbox

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[
        bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1],
        bbox[3][0]:bbox[3][1]
        ] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.float32), isVector=True)
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.float32))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)

    return seg_old_size_postprocessed.astype(np.float32)