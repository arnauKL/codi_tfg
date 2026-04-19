import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    CenterSpatialCropd, NormalizeIntensityd, Lambdad
)

def get_3d_transforms(roi_size=(76, 76, 76)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        NormalizeIntensityd(keys=["image"]),
    ])

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