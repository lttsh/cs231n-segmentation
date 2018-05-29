import torch
from code.inpainting import PartialConv2d
import numpy as np

convLayer = PartialConv2d(1, 1, 2)
features = np.array([[1,2,3], [4,5,6], [7,8,9]])
mask = np.array([[0,1,0],[0,0,0],[1,1,0]])
mask = np.reshape(mask, (1, 1, 3, 3))
features = np.reshape(features, (1, 1, 3, 3))

tensor_x = torch.from_numpy(features).type(torch.float32)
tensor_mask = torch.from_numpy(mask).type(torch.float32)


print(tensor_x)
print (tensor_mask)
result, mask = convLayer(tensor_x, tensor_mask)

print(result, mask)
