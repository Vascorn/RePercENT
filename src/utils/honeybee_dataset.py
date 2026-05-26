from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
import random
import copy
import torch
import matplotlib.pyplot as plt
import argparse
import os


def load_honeybee_dataset():
    # Clinical data embeddings - We select the Qwen embeddings
    clinical_qwen = load_dataset("Lab-Rasool/TCGA", "clinical", split="qwen")

    # Pathology report embeddings - We select the Qwen embeddings
    pathology_qwen = load_dataset("Lab-Rasool/TCGA", "pathology_report", split="qwen")

    # Whole slide image embeddings 
    wsi_dataset = load_dataset("Lab-Rasool/TCGA", "wsi", split="uni")

    # Molecular data embeddings
    molecular_dataset = load_dataset("Lab-Rasool/TCGA", "molecular", split="senmo")

    # Radiology embeddings
    radiology_remedis = load_dataset("Lab-Rasool/TCGA", "radiology", split="remedis")


    return {
        "clinical_qwen": clinical_qwen,
        "pathology_qwen": pathology_qwen,
        "wsi": wsi_dataset,
        "molecular": molecular_dataset,
        "radiology_remedis": radiology_remedis,
    }


mod_identifiers = {"clinical_qwen": "case_submitter_id",
                    "molecular": "PatientID",
                    "pathology_qwen": "PatientID",
                    "radiology_remedis": "PatientID",
                    "wsi": "PatientID"}


class TCGAEmbeddingsExtractor:
    """Load and align modality embeddings by patient id."""

    def __init__(
        self,
        hf_data,
        modalities=None,
        mod_identifiers=None,
        id_normalizers=None,
        require_all_modalities=False,
    ):
        self.hf_dataset = hf_data
        self.modalities = modalities or [
            "clinical_qwen",
            "pathology_qwen",
            "wsi",
            "molecular",
            "radiology_remedis",
        ]

        if mod_identifiers is None:
            raise ValueError("mod_identifiers must be provided")

        self.mod_identifiers = {m: mod_identifiers[m] for m in self.modalities}
        self.id_normalizers = id_normalizers or {}
        self.require_all_modalities = require_all_modalities
        self.index = {}
        self.modality_patient_ids = {}

        self._build_index()

        if self.require_all_modalities:
            self.patient_ids = sorted(set.intersection(*self.modality_patient_ids.values()))
            print(f"Number of aligned patients: {len(self.patient_ids)}")
        else:
            self.patient_ids = sorted(self.index.keys())

        self.get_stats()

    def _make_patient_record(self):
        return {m: [] for m in self.modalities}

    def _normalize_patient_id(self, modality, raw_id):
        fn = self.id_normalizers.get(modality)
        return fn(raw_id) if fn is not None else raw_id

    def _extract_embedding(self, modality, row):
        modality_l = modality.lower()

        if "molecular" in modality_l:
            emb = row["Embeddings"]
        else:
            emb = row["embedding"] if "embedding" in row else row["Embeddings"]

        if isinstance(emb, (bytes, bytearray, memoryview)):
            shape = tuple(row["embedding_shape"])
            return np.frombuffer(emb, dtype=np.float32).reshape(shape)

        return np.asarray(emb, dtype=np.float32)

    def _build_index(self):
        if self.index:
            print("Index already built, skipping rebuild.")
            return

        print("Building patient index...")

        to_float_or_none = lambda v: None if v is None else float(v)

        for modality in self.modalities:
            ds = self.hf_dataset[modality]
            id_col = self.mod_identifiers[modality]
            seen = set()

            for row in ds:
                raw_id = row[id_col]
                pid = self._normalize_patient_id(modality, raw_id)
                seen.add(pid)

                if pid not in self.index:
                    self.index[pid] = self._make_patient_record()

                value = self._extract_embedding(modality, row)
                self.index[pid][modality].append(value)

                if "clinical" in modality.lower():
                    self.index[pid]["cancer_type"] = row.get("project_id", None)
                    self.index[pid]["vital_status"] = row.get("vital_status", None)
                    self.index[pid]["days_to_death"] = to_float_or_none(row.get("days_to_death", None))
                    self.index[pid]["days_to_last_follow_up"] = to_float_or_none(row.get("days_to_last_follow_up", None))
                    self.index[pid]["days_to_diagnosis"] = to_float_or_none(row.get("days_to_diagnosis", None))

            self.modality_patient_ids[modality] = seen
            print(f"  {modality}: {len(seen)} unique patients")

    def get_stats(self, plot_distributions=False, **kwargs):
        self.stats = {}
        all_wsi_patch_counts = []
        for pid in self.patient_ids:
            patient_record = self.index[pid]
            self.stats[pid] = {mod: len(patient_record[mod]) for mod in self.modalities}
            if "wsi" in self.modalities:
                slide_patch_counts = [int(emb.shape[0]) for emb in patient_record["wsi"]]
                self.stats[pid]["wsi_patch_counts"] = slide_patch_counts
                all_wsi_patch_counts.extend(slide_patch_counts)

        self.stats["total_patients"] = len(self.patient_ids)
        self.stats["max_embeddings_per_modality"] = {
            mod: max([self.stats[pid][mod] for pid in self.patient_ids], default=0)
            for mod in self.modalities
        }
        self.stats["mean_embeddings_per_modality"] = {
            mod: float(np.mean([self.stats[pid][mod] for pid in self.patient_ids])) if self.patient_ids else 0.0
            for mod in self.modalities
        }
        self.stats["distribution_embeddings_per_modality"] = {}

        for mod in self.modalities:
            sample_counts = [self.stats[pid][mod] for pid in self.patient_ids]
            unique_counts, frequencies = np.unique(sample_counts, return_counts=True)
            self.stats["distribution_embeddings_per_modality"][mod] = {
                int(count): int(freq)
                for count, freq in zip(unique_counts, frequencies)
            }

        if "wsi" in self.modalities and all_wsi_patch_counts:
            unique_counts, frequencies = np.unique(all_wsi_patch_counts, return_counts=True)
            self.stats["max_wsi_patch_count"] = int(max(all_wsi_patch_counts))
            self.stats["mean_wsi_patch_count"] = float(np.mean(all_wsi_patch_counts))
            self.stats["distribution_wsi_patch_count"] = {
                int(count): int(freq)
                for count, freq in zip(unique_counts, frequencies)
            }

        print(f"in stats max embeddings per modality: {self.stats['max_embeddings_per_modality']}")
        print(f"in stats mean embeddings per modality: {self.stats['mean_embeddings_per_modality']}")
        print(f"in stats distribution embeddings per modality: {self.stats['distribution_embeddings_per_modality']}")
        if "wsi" in self.modalities and all_wsi_patch_counts:
            print(f"in stats max WSI patch count: {self.stats['max_wsi_patch_count']}")
            print(f"in stats mean WSI patch count: {self.stats['mean_wsi_patch_count']}")
            print(f"in stats distribution WSI patch count: {self.stats['distribution_wsi_patch_count']}")

        if plot_distributions:
            self.plot_sample_count_distributions(save_path=kwargs.get("save_path", "./sample_count_distributions.png"))

        return self.stats

    def plot_sample_count_distributions(self, save_path=None, show=True):
        if not hasattr(self, "stats") or "distribution_embeddings_per_modality" not in self.stats:
            self.get_stats(plot_distributions=False)

        num_modalities = len(self.modalities)
        fig, axes = plt.subplots(num_modalities, 1, figsize=(8, max(3, 3 * num_modalities)))

        if num_modalities == 1:
            axes = [axes]

        for ax, modality in zip(axes, self.modalities):
            distribution = self.stats["distribution_embeddings_per_modality"][modality]
            x = sorted(distribution.keys())
            y = [distribution[val] for val in x]
            ax.bar(x, y)
            ax.set_title(f"{modality} sample-count distribution")
            ax.set_xlabel("Number of embeddings per patient")
            ax.set_ylabel("Number of patients")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def to_dataset(self, wsi_embedding_mode="patch", wsi_pooling_method="mean"):
        return MultimodalTCGA(
            embeddings_dict=self.index,
            patient_ids=self.patient_ids,
            modalities=self.modalities,
            stats=copy.deepcopy(self.stats),
            wsi_embedding_mode=wsi_embedding_mode,
            wsi_pooling_method=wsi_pooling_method,
        )


class MultimodalTCGA(Dataset):
    """Padded dataset built from pre-extracted embeddings."""

    def __init__(
        self,
        embeddings_dict,
        patient_ids=None,
        modalities=None,
        stats=None,
        wsi_embedding_mode="patch",
        wsi_pooling_method="mean",
    ):
        self.index = embeddings_dict
        self.patient_ids = patient_ids or sorted(embeddings_dict.keys())
        self.modalities = modalities or [
            key for key in next(iter(embeddings_dict.values())).keys() if key != "cancer_type"
        ]
        self.stats = copy.deepcopy(stats) if stats is not None else None
        self.wsi_embedding_mode = wsi_embedding_mode
        self.wsi_pooling_method = wsi_pooling_method

        if self.wsi_embedding_mode not in {"slide", "patch"}:
            raise ValueError("wsi_embedding_mode must be either 'slide' or 'patch'")

        self.calc_input_shapes()

    def get_stats(self, *args, **kwargs):
        if self.stats is not None:
            return self.stats

        self.stats = {}
        all_wsi_patch_counts = []
        for pid in self.patient_ids:
            patient_record = self.index[pid]
            self.stats[pid] = {mod: len(patient_record[mod]) for mod in self.modalities}
            if "wsi" in self.modalities:
                slide_patch_counts = [int(emb.shape[0]) for emb in patient_record["wsi"]]
                self.stats[pid]["wsi_patch_counts"] = slide_patch_counts
                all_wsi_patch_counts.extend(slide_patch_counts)

        self.stats["max_embeddings_per_modality"] = {
            mod: max([self.stats[pid][mod] for pid in self.patient_ids], default=0)
            for mod in self.modalities
        }
        if "wsi" in self.modalities and all_wsi_patch_counts:
            self.stats["max_wsi_patch_count"] = int(max(all_wsi_patch_counts))
        else:
            self.stats["max_wsi_patch_count"] = 0
        return self.stats

    def pool_embeddings(self, embeddings, pooling_method="mean"):
        match pooling_method:
            case "mean":
                return torch.mean(embeddings, dim=0)
            case "max":
                return torch.max(embeddings, dim=0).values
            case "sum":
                return torch.sum(embeddings, dim=0)
            case _:
                raise ValueError(f"Unsupported pooling method: {pooling_method}")

    def augment_embeddings(self, embedding, embedding_type, **kwargs):
        match embedding_type:
            case "clinical_qwen" | "pathology_qwen" | "molecular":
                noise = kwargs.get("noise", 0.01)
                augmented = embedding + np.random.normal(0, noise, embedding.shape)
                return torch.from_numpy(augmented.astype(np.float32))
            case "wsi":
                prob = kwargs.get("drop_prob", 0.1)
                aug_embedding = np.array(embedding, dtype=np.float32, copy=True)
                selected_patches = np.random.choice(
                    [True, False],
                    size=aug_embedding.shape[0],
                    p=[1 - prob, prob],
                )

                if self.wsi_embedding_mode == "patch":
                    aug_embedding[~selected_patches] = 0.0
                else:
                    aug_embedding = aug_embedding[selected_patches]
                    if aug_embedding.shape[0] == 0:
                        aug_embedding = np.zeros((1, embedding.shape[-1]), dtype=np.float32)

                return torch.from_numpy(aug_embedding.astype(np.float32))
            case _:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")

    def __len__(self):
        return len(self.patient_ids)

    def calc_input_shapes(self):
        stats = self.get_stats()
        self.input_shapes = copy.deepcopy(stats["max_embeddings_per_modality"])
        if "wsi" in self.modalities and self.wsi_embedding_mode == "patch":
            self.input_shapes["wsi_patches"] = stats.get("max_wsi_patch_count", 0)
        self.embedding_dims = self._infer_embedding_dims()
        print(f"Calculated input shapes per modality: {self.input_shapes}")
        print(f"Calculated embedding dimensions per modality: {self.embedding_dims}")

    def _infer_embedding_dims(self):
        embedding_dims = {}
        for modality in self.modalities:
            embedding_dims[modality] = None
            for pid in self.patient_ids:
                embeddings = self.index[pid][modality]
                if len(embeddings) == 0:
                    continue

                first_embedding = self._to_float_tensor(embeddings[0])
                embedding_dims[modality] = int(first_embedding.shape[-1])
                break

            if embedding_dims[modality] is None:
                raise ValueError(f"Could not infer embedding dimension for modality '{modality}'")

        return embedding_dims

    def _to_float_tensor(self, embedding):
        if isinstance(embedding, torch.Tensor):
            return embedding.to(dtype=torch.float32)
        return torch.as_tensor(embedding, dtype=torch.float32)

    def _prepare_wsi_embeddings(self, embeddings, augmented_embeddings):
        if self.wsi_embedding_mode == "slide":
            pooled_embeddings = torch.stack(
                [self.pool_embeddings(self._to_float_tensor(emb), self.wsi_pooling_method) for emb in embeddings],
                dim=0,
            )
            pooled_aug_embeddings = torch.stack(
                [self.pool_embeddings(self._to_float_tensor(aug_emb), self.wsi_pooling_method) for aug_emb in augmented_embeddings],
                dim=0,
            )
            return pooled_embeddings, pooled_aug_embeddings

        patch_embeddings, patch_pad_masks = [], []
        patch_aug_embeddings = []
        for emb, aug_emb in zip(embeddings, augmented_embeddings):
            temp_patch_emb, temp_pad_mask = self._pad_modality_tensor(
                self._to_float_tensor(emb),
                self.input_shapes["wsi_patches"],
            )
            patch_embeddings.append(temp_patch_emb)
            patch_pad_masks.append(temp_pad_mask)

            temp_aug_patch_emb, _ = self._pad_modality_tensor(
                self._to_float_tensor(aug_emb),
                self.input_shapes["wsi_patches"],
            )
            patch_aug_embeddings.append(temp_aug_patch_emb)

        patch_embeddings = torch.stack(patch_embeddings, dim=0)
        patch_pad_mask = torch.stack(patch_pad_masks, dim=0)
        patch_aug_embeddings = torch.stack(patch_aug_embeddings, dim=0)

        return (patch_embeddings, patch_pad_mask), patch_aug_embeddings

    def _pad_modality_tensor(self, embeddings, target_length):
        current_length = embeddings.shape[0]
        pad_mask = torch.zeros(target_length, dtype=torch.bool)
        pad_mask[:current_length] = True

        if current_length == target_length:
            return embeddings, pad_mask

        padded = torch.zeros((target_length, *embeddings.shape[1:]), dtype=embeddings.dtype)
        padded[:current_length] = embeddings
        return padded, pad_mask

    def _metadata_value(self, pid, key, fallback= -1):
        value = self.index[pid].get(key, fallback)
        return fallback if value is None else value

    def _empty_modality_sample(self, modality):
        target_length = self.input_shapes[modality]
        feature_dim = self.embedding_dims[modality]

        if modality == "wsi":
            if self.wsi_embedding_mode == "slide":
                empty_embeddings = torch.zeros((target_length, feature_dim), dtype=torch.float32)
                empty_aug_embeddings = torch.zeros((target_length, feature_dim), dtype=torch.float32)
                empty_pad_mask = torch.zeros(target_length, dtype=torch.bool)
            else:
                patch_length = self.input_shapes["wsi_patches"]
                empty_embeddings = torch.zeros((target_length, patch_length, feature_dim), dtype=torch.float32)
                empty_aug_embeddings = torch.zeros((target_length, patch_length, feature_dim), dtype=torch.float32)
                empty_pad_mask = torch.zeros((target_length, patch_length), dtype=torch.bool)
        else:
            empty_embeddings = torch.zeros((target_length, feature_dim), dtype=torch.float32)
            empty_aug_embeddings = torch.zeros((target_length, feature_dim), dtype=torch.float32)
            empty_pad_mask = torch.zeros(target_length, dtype=torch.bool)

        return (empty_embeddings, empty_aug_embeddings, empty_pad_mask, False)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        sample = {
            "patient_id": pid,
            "cancer_type": self._metadata_value(pid, "cancer_type", "unknown"),
            "vital_status": self._metadata_value(pid, "vital_status", "unknown"),
            "days_to_death": self._metadata_value(pid, "days_to_death", -1.0),
            "days_to_last_follow_up": self._metadata_value(pid, "days_to_last_follow_up", -1.0),
            "days_to_diagnosis": self._metadata_value(pid, "days_to_diagnosis", -1.0),
        }

        for modality in self.modalities:
            embeddings = self.index[pid][modality]

            if len(embeddings) == 0:
                sample[modality] = self._empty_modality_sample(modality)
                continue

            aug_embeddings = [self.augment_embeddings(emb, modality) for emb in embeddings]

            if modality == "wsi":
                embeddings, aug_embeddings = self._prepare_wsi_embeddings(embeddings, aug_embeddings)
                if self.wsi_embedding_mode == "slide":
                    embeddings, pad_mask = self._pad_modality_tensor(embeddings, self.input_shapes[modality])
                    aug_embeddings, _ = self._pad_modality_tensor(aug_embeddings, self.input_shapes[modality])
                    sample[modality] = (embeddings, aug_embeddings, pad_mask, True)
                else:
                    patch_embeddings, patch_pad_mask = embeddings
                    padded_patch_embeddings, _ = self._pad_modality_tensor(patch_embeddings, self.input_shapes["wsi"])
                    padded_aug_patch_embeddings, _ = self._pad_modality_tensor(aug_embeddings, self.input_shapes["wsi"])
                    padded_patch_pad_mask, _ = self._pad_modality_tensor(patch_pad_mask, self.input_shapes["wsi"])
                    sample[modality] = (padded_patch_embeddings, padded_aug_patch_embeddings, padded_patch_pad_mask, True)
            else:
                embeddings = torch.stack([self._to_float_tensor(emb) for emb in embeddings], dim=0)
                aug_embeddings = torch.stack([self._to_float_tensor(aug_emb) for aug_emb in aug_embeddings], dim=0)
                embeddings, pad_mask = self._pad_modality_tensor(embeddings, self.input_shapes[modality])
                aug_embeddings, _ = self._pad_modality_tensor(aug_embeddings, self.input_shapes[modality])
                sample[modality] = (embeddings, aug_embeddings, pad_mask, True)

        return sample



def main():
    parser = argparse.ArgumentParser(description="Creating the MultimodalTCGA dataset")
    parser.add_argument("--plot_distributions", action="store_true", help="Whether to plot sample count distributions")
    parser.add_argument("--data_save_path", type=str, default="./data/honeybee/datasets/", help="Path to save the created dataset")
    parser.add_argument("--modalities", nargs="+", default=["clinical_qwen", "pathology_qwen", "wsi", "molecular"], help="Modalities to include in the dataset")
    parser.add_argument("--wsi_embedding_mode", type=str, choices=["slide", "patch"], default="slide", help="How to handle WSI embeddings: 'slide' for pooling to slide-level, 'patch' for keeping patch-level with padding")


    args = parser.parse_args()


    datasets = load_honeybee_dataset()
    
    # Example: Access embeddings
    for index, item in enumerate(datasets["clinical_qwen"]):
        embedding = np.frombuffer(item.get("embedding"), dtype=np.float32).reshape(item.get("embedding_shape"))
        print(f"Clinical Qwen embedding shape: {embedding.shape}") 
        break

    for index, item in enumerate(datasets["pathology_qwen"]):
        embedding = np.frombuffer(item.get("embedding"), dtype=np.float32).reshape(item.get("embedding_shape"))
        print(f"Pathology Qwen embedding shape: {embedding.shape}") 
        break

    for index, item in enumerate(datasets["wsi"]):
        embedding = np.frombuffer(item.get("embedding"), dtype=np.float32).reshape(item.get("embedding_shape"))
        print(f"WSI embedding shape: {embedding.shape}") 
        break
    
    for index, item in enumerate(datasets["molecular"]):
        embedding = np.asarray(item.get("Embeddings")).reshape(item.get("embedding_shape"))
        print(f"Molecular embedding shape: {embedding.shape}") 
        break

    
    # First extract and align the raw embeddings, then build the minimal padded dataset.
    extractor = TCGAEmbeddingsExtractor(
        datasets,
        mod_identifiers=mod_identifiers,
        modalities=["clinical_qwen", "pathology_qwen", "wsi", "molecular"],
        require_all_modalities=True
    )
    mulitmodal_tcga_data = extractor.to_dataset(wsi_embedding_mode=args.wsi_embedding_mode)

    print(f"Unique patient IDs in MultimodalTCGA dataset: {len(mulitmodal_tcga_data.patient_ids)}")
    extractor.get_stats(plot_distributions=args.plot_distributions)

    # Save the lightweight padded dataset instead of the full HF-backed extractor.
    os.makedirs(args.data_save_path, exist_ok=True)
    torch.save(mulitmodal_tcga_data, os.path.join(args.data_save_path, f"dataset_01_{args.wsi_embedding_mode}.pt"))

   
    # Tesing loading the saved dataset
    load_path = os.path.join(args.data_save_path, f"dataset_01_{args.wsi_embedding_mode}.pt")
    dataset = torch.load(load_path, weights_only=False)
    
     #Example of accessing a batch of data
    # NOTE: The "patch" yields a 2D padding mask of shape (num_slide, num_patches) while the "slide" mode yields a 1D padding mask of shape (num_slide,)
    loader = torch.utils.data.DataLoader(dataset, batch_size= 1, shuffle=True)
    print(f"dataset length: {len(dataset)}")
    
    for batch in loader:
        print(f"Batch patient IDs: {batch['patient_id']}")
        print(f"Batch cancer types: {batch['cancer_type']}")
        print(f"Batch vital status: {batch['vital_status']}")
        print(f"Batch days to death: {batch['days_to_death']}")
        print(f"Batch days to last follow up: {batch['days_to_last_follow_up']}")
        print(f"Batch days to diagnosis: {batch['days_to_diagnosis']}")

        for modality in mulitmodal_tcga_data.modalities:
            embeddings, aug_embeddings, pad_mask, has_data = batch[modality]
            if modality == "wsi":
                # check if the augmented and original embeddings are identical
                print(f"Original and augmented WSI embeddings identical: {torch.equal(embeddings, aug_embeddings)}")
            print(f"Modality: {modality},\nEmbeddings (shape: {embeddings.shape}): {embeddings}\nAugmented Embeddings (shape: {aug_embeddings.shape}): {aug_embeddings}\nPad Mask (shape: {pad_mask.shape}): {pad_mask}\nHas Data: {has_data}\n")

        break

if __name__ == "__main__":
    main()