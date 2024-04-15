import nibabel as nib

img = nib.load('../Data/atlases/components_1024_task.nii.gz')

print(img.affine)
print(img.shape)