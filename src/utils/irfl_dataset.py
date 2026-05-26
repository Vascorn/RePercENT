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



def make_dataset(total_data: Dict[str, Any]= None, data_type: Literal['train', 'test'] = 'train', include_original: bool = False, num_modalities: int = 3) -> Dataset:
    """
    Create a IRFLDataset instance from the provided dictionary data.

    Args:
        total_data: A dictionary containing all the data samples including images, texts, definitions embeddings and distractors (if test data).
        data_type: 'train' or 'test' to indicate the type of data. The main difference is that the test data frame also contains distractors, which are not provided in the train data frame.
        include_original: Whether to include the original inputs (images, phrases, definitions, distractors) in the dataset samples. Note that the distractors are only available for test data.
        num_modalities: Number of modalities to use. The default is 3, corresponding to images, phrases, and definitions.
    Returns:
        An instance of IRFLDataset containing the provided data.
    """

    prefix = f"{data_type}_"
    
    
    data = {
        "images": total_data.get(prefix + "images", None) if data_type == 'train' else total_data.get(prefix + "answers", None), # for test data, the "answers" key contains the original images, while the "images" key contains the images for the train.
        "texts": total_data[prefix + "phrases"],
        "pad_masks": total_data[prefix + "phrases_mask"],
        "definitions": total_data.get(prefix + "definitions", None), # optional, only if num_modalities == 3
        "definitions_mask": total_data.get(prefix + "definitions_mask", None),
        "distractors": total_data.get(prefix + "distractors", None), # present only in test data

        # Augmented data
        "images_aug": total_data.get(prefix + "images_aug", None) if data_type == 'train' else total_data.get(prefix + "answers_aug", None), # for test data, the "answers_aug" key contains the augmented images, while the "images_aug" key contains the augmented images for the train.
        "texts_aug": total_data[prefix + "phrases_aug"],
        "pad_masks_aug": total_data[prefix + "phrases_mask_aug"],
        "definitions_aug": total_data.get(prefix + "definitions_aug", None), # optional, only if num_modalities == 3
        "definitions_mask_aug": total_data.get(prefix + "definitions_mask_aug", None),
        "figurative_type": total_data.get(prefix + "figurative_type", None)
    }


    # Original (non-augmented) inputs
    if include_original:
        data.update({
            "in_images": total_data[prefix + "images_in"],
            "in_phrases": total_data[prefix + "phrases_in"],
            "in_definitions": total_data[prefix + "definitions_in"],
            "in_distractors": total_data.get(prefix + "images_distractors_in", None)
        })



    return IRFLDataset(total_data=data, data_type=data_type, num_modalities=num_modalities, include_original=include_original), data

    

class IRFLDataset(Dataset):
    def __init__(self, total_data, data_type: Literal['train', 'test'] = 'train', num_modalities: int = 3, include_original: bool = True, sample_one_aug: bool = True):
        """
        total_data: A dictionary containing all the data samples including images, texts, definitions embeddings and distractors (if test data).
        data_type: 'train' or 'test' to indicate the type of data. The main difference is that the test data frame also contains distractors, which are not provided in the train data frame.
        self.num_modalities: Number of modalities to use. The default is 2, correspoding to the image and the relevant figurative language text. The third modality is optional and is the definition text.
        """
        self.total_data = total_data
        self.data_type = data_type
        self.num_modalities = num_modalities
        self.include_original = include_original
        self.sample_one_aug = sample_one_aug


        self.in_images = total_data.get('in_images', None) # optional, just to keep track of the original images
        self.in_phrases = total_data.get('in_phrases', None) # optional, just to keep track of the original phrases
        self.in_definitions = total_data.get('in_definitions', None) # optional, just to keep track of the original definitions
        self.in_distractors = total_data.get('in_distractors', None) # optional, just to keep track of the original distractors if the data_type is 'test'

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
        
        self.figurative_type = total_data.get('figurative_type', None) # only if data_type == 'test'

    def __len__(self):
        return len(self.images)


    def _to_f32(self, x):
        return x.to(torch.float32) if torch.is_tensor(x) else x

    def _pick_aug(self, aug_item):
        """
        If aug_item is:
          - Tensor: return either itself or one slice if it has an aug dimension
          - List[Tensor]: pick one tensor
        """

        if aug_item is None:
            print(f"aug_item is None")
            return None

        # tensor of augmentations per sample (e.g., [n_aug, ...])
        if torch.is_tensor(aug_item) and self.sample_one_aug and aug_item.ndim >= 1 and aug_item.shape[0] > 1:
            j = torch.randint(0, aug_item.shape[0], (1,)).item()
            return aug_item[j], j



    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x = {
            "images": self._to_f32(self.images[idx]),
            "texts": self._to_f32(self.texts[idx]),
            "pad_masks": self.pad_masks[idx],
        }
        
        if self.num_modalities == 3:
            if self.definitions is None or self.definitions_mask is None:
                raise ValueError("num_modalities=3 but definitions/definitions_mask are missing.")
            x["definitions"] = self._to_f32(self.definitions[idx])
            x["definitions_mask"] = self.definitions_mask[idx]

        # Augmented sample (optional but consistent)
        x_aug = None
        if self.images_aug is not None and self.texts_aug is not None and self.pad_masks_aug is not None:
            
            picked_image_aug, image_aug_id = self._pick_aug(self.images_aug[idx])
            picked_text_aug, text_aug_id = self._pick_aug(self.texts_aug[idx])
            x_aug = {
                "images": self._to_f32(picked_image_aug),
                "texts": self._to_f32(picked_text_aug),
                "pad_masks": self.pad_masks_aug[idx][text_aug_id],
            }
            if self.num_modalities == 3 and self.definitions_aug is not None and self.definitions_mask_aug is not None:
                picked_def_aug, def_aug_id = self._pick_aug(self.definitions_aug[idx])
                x_aug["definitions"] = self._to_f32(picked_def_aug)
                x_aug["definitions_mask"] = self.definitions_mask_aug[idx][def_aug_id] 
        
        out: Dict[str, Any] = {"x": x, "x_aug": x_aug}

        # Test-only extras
        if self.data_type == "test":
            out["distractors"] = self.distractors[idx]
            out["figurative_type"] = self.figurative_type[idx]

        # Originals only if requested AND available
        if self.include_original:
            out["orig"] = {
                "images": None if self.in_images is None else self.in_images[idx],
                "phrases": None if self.in_phrases is None else self.in_phrases[idx],
                "definitions": None if self.in_definitions is None else self.in_definitions[idx]
            }
            if self.data_type == "test":
                
                out["orig"]["distractors"] = self.in_distractors[idx]
        
        return out

    def sample_batch(self, batch_size: int):
        idxs = np.random.choice(len(self), batch_size, replace=False)
        return [self[i] for i in idxs]