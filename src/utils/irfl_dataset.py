import torch
import numpy as np
from torch.utils.data import Dataset
# Adjust sys.path to import always from src
import os
import sys
import torch
from typing import Dict, Any, Callable, Optional, Sequence, Tuple, List, Union, Literal
import torch.nn as non

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# def save_dataset(dataset, save_path: str, data_config: Dict[str, Any]= None):
#     # create directory if it doesn't exist
    
#     os.makedirs(os.path.dirname(save_path + "/dataset.pt"), exist_ok=True)
#     torch.save(dataset, os.path.join(save_path, "dataset.pt"))
#     print(f"Dataset saved at {save_path}")
#     # create a README file with the data configuration
#     if data_config is not None:
#         readme_path = os.path.join(save_path, "README.md")
#         with open(readme_path, 'w') as f:
#                 f.write("### Dataset Configuration\n\n")
#                 for key, value in data_config["create_data"].items():
#                     if isinstance(value, dict):
#                         print(f"Writing config section: {key}")
#                         f.write(f"* {key}: \n\n")
#                         for sub_key, sub_value in value.items():
#                             f.write(f"  - **{sub_key}**: {sub_value}\n")
#                         f.write("\n")
#                     else:
#                         f.write(f"* **{key}**: {value}\n")
#         print(f"README saved at {readme_path}")
#     return

# def save_data_split(train_dataset, test_dataset, save_path: str, split_id: str= "0"):
#     os.makedirs(save_path, exist_ok=True)
#     torch.save({'train_dataset': train_dataset, 'test_dataset': test_dataset}, os.path.join(save_path, f"data_split_{split_id}.pt"))
#     print(f"Data split saved at {save_path}")
#     return

class IRFLDataset(Dataset):
    def __init__(self, total_data, data_type: Literal['train', 'test'] = 'train', num_modalities: int = 2):
        """
        total_data: A dictionary containing all the data samples including images, texts, definitions embeddings and distractors (if test data).
        data_type: 'train' or 'test' to indicate the type of data. The main difference is that the test data frame also contains distractors, which are not provided in the train data frame.
        self.num_modalities: Number of modalities to use. The default is 2, correspoding to the image and the relevant figurative language text. The third modality is optional and is the definition text.
        """
        self.total_data = total_data
        self.data_type = data_type
        self.num_modalities = num_modalities

        self.images = total_data['images']
        self.texts = total_data['texts']
        self.pad_masks = total_data['pad_masks']

        self.definitions = total_data.get('definitions', None) # optional, only if num_modalities == 3
        self.definitions_mask = total_data.get('definitions_mask', None) # optional, only if num_modalities == 3
        self.distractors = total_data.get('distractors', None) # only for test data

        self.text_shape = self.texts.shape[1:]
        self.image_shape = self.images.shape[1:]

        # initialize also the Augmented data
        # NOTE: per image, text, definition there could be multiple augmentations so that one can sample from them during training
        self.images_aug = total_data.get('images_aug', None)
        self.texts_aug = total_data.get('texts_aug', None)
        self.pad_masks_aug = total_data.get('pad_masks_aug', None)

        #NOTE: The definitions_aug and definitions_mask_aug are a list of tensors, each tensor corresponds to multiple definition 
        # augmentation per sample. The list is used as there are different number of augmentations per sample.
        self.definitions_aug = total_data.get('definitions_aug', None) # optional, only if num_modalities == 3
        self.definitions_mask_aug = total_data.get('definitions_mask_aug', None) # optional, only if num_modalities == 3


        self.text_shape_aug = self.texts_aug.shape[1:] if self.texts_aug is not None else None
        self.image_shape_aug = self.images_aug.shape[1:] if self.images_aug is not None else None
        

    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        match self.data_type:
            case 'train':
                return self._get_train_item(idx)
            case 'test':
                return self._get_test_item(idx)
            case _:
                raise ValueError(f"Invalid data_type: {self.data_type}. Must be 'train' or 'test'.")

    def _get_train_item(self, idx):
        if self.num_modalities == 2:
            return [self.images[idx].to(torch.float32), self.texts[idx].to(torch.float32), self.pad_masks[idx].to(torch.float32)], \
                [self.images_aug[idx].to(torch.float32), self.texts_aug[idx].to(torch.float32), self.pad_masks_aug[idx].to(torch.float32)]  # Augmented part
        elif self.num_modalities == 3:
            return [self.images[idx].to(torch.float32), self.texts[idx].to(torch.float32), self.pad_masks[idx].to(torch.float32), self.definitions[idx].to(torch.float32), self.definitions_mask[idx].to(torch.float32)], \
                [self.images_aug[idx].to(torch.float32), self.texts_aug[idx].to(torch.float32), self.pad_masks_aug[idx].to(torch.float32), self.definitions_aug[idx].to(torch.float32), self.definitions_mask_aug[idx].to(torch.float32)] # Augmented part
        
    def _get_test_item(self, idx):
        if self.num_modalities == 2:
            return [self.images[idx].to(torch.float32), self.texts[idx].to(torch.float32), self.pad_masks[idx].to(torch.float32), self.distractors[idx]],\
                [self.images_aug[idx].to(torch.float32), self.texts_aug[idx].to(torch.float32), self.pad_masks_aug[idx].to(torch.float32)]  # Augmented part
        elif self.num_modalities == 3:
            return [self.images[idx].to(torch.float32), self.texts[idx].to(torch.float32), self.pad_masks[idx].to(torch.float32), self.definitions[idx].to(torch.float32), self.definitions_mask[idx].to(torch.float32), self.distractors[idx]], \
                [self.images_aug[idx].to(torch.float32), self.texts_aug[idx].to(torch.float32), self.pad_masks_aug[idx].to(torch.float32), self.definitions_aug[idx].to(torch.float32), self.definitions_mask_aug[idx].to(torch.float32)] # Augmented part

        
    def sample_batch(self, batch_size):
        sample_idxs = np.random.choice(self.__len__(), batch_size, replace=False)
        samples = self.__getitem__(sample_idxs)
        return samples