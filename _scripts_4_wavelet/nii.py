import nibabel as nib
import numpy as np

fp = "E:/LD-CT SR/Data/Image_NC-CT/s0073351_0000.nii.gz"
nii = nib.load(fp)
arr = nii.get_fdata()

print("min:", arr.min())
print("max:", arr.max())
print("mean:", arr.mean())
print("std:", arr.std())
