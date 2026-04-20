import torch
from monai.transforms import (
    Compose, LoadImaged, 
    EnsureChannelFirstd, 
    CenterSpatialCropd, 
    NormalizeIntensityd, 
    Lambdad,
    ResizeWithPadOrCropd,
    Orientationd
)



def get_3d_transforms(roi_size=(76, 76, 76)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        NormalizeIntensityd(keys=["image"]),
    ])


def get_3d_padding_cropping_transforms(spatial_size):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Standardize orientation first
        Orientationd(keys=["image"], axcodes="RAS"), 
        # This handles BOTH cases: pads if < 128, crops if > 128
        ResizeWithPadOrCropd(
            keys=["image"], 
            spatial_size=spatial_size, 
            mode="constant"
        ),
        NormalizeIntensityd(keys=["image"]),
    ])


############# 2D


def sum_slices(data):
    return torch.sum(data, dim=-1)


def get_2d_sum_transforms(roi_size=(76, 76, 76)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        Lambdad(keys=["image"], func=sum_slices),
        NormalizeIntensityd(keys=["image"]),
    ])


# (test): Only sum slices 30 to 45 where the striatum usually lives
def sum_striatum_only(data):
    # this is just an idea, Im not even calling this function for now
    return torch.sum(data[:, :, :, 30:45], dim=-1) # q això realment podrien ser percentils/ proporcions de la imatge

def get_2d_sum_striatum_transforms(roi_size=(76, 76, 76)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        Lambdad(keys=["image"], func=sum_striatum_only),
        NormalizeIntensityd(keys=["image"]),
    ])


def get_2d_sum_transforms_padding(spatial_size):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Standardize orientation first
        Orientationd(keys=["image"], axcodes="RAS"), 
        # This handles BOTH cases: pads if < 128, crops if > 128
        ResizeWithPadOrCropd(
            keys=["image"], 
            spatial_size=spatial_size, 
            mode="constant"
        ),
        Lambdad(keys=["image"], func=sum_slices),
        NormalizeIntensityd(keys=["image"]),
    ])