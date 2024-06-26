# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import torch

def merge_checkpoints(checkpoint_path, num_parts=5):
    combined_state_dict = {}
    for i in range(1, num_parts + 1):
        part_path = checkpoint_path.replace('.pt', '_part{}.pt'.format(i))
        part_checkpoint = torch.load(part_path)
        part_state_dict = part_checkpoint['model_state_dict']
        combined_state_dict.update(part_state_dict)

    full_checkpoint = {'model_state_dict': combined_state_dict}
    torch.save(full_checkpoint, checkpoint_path)
    print('merging {}: finished'.format(checkpoint_path))

merge_checkpoints('foundation.pt', num_parts=5)
merge_checkpoints('chat.pt', num_parts=5)
