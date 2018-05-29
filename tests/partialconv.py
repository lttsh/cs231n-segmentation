import torch
from code.inpainting import PartialConv2d


convLayer = PartialConv2d(1, 1, 2)
features = np.array([[1,2,3], [4,5,6], [7,8,9]])
mask = np.array([[0,1,0],[0,0,0],[1,1,0]])
tensor_x = torch.from_numpy(features)
tensor_mask = torch.from_numpy(mask)

print(tensor_x)
