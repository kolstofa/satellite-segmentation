import torch

if torch.cuda.is_available():
    print(torch.cuda.device_count(),
          torch.cuda.current_device(),
          torch.cuda.get_device_name())
else: print('cuda is died')