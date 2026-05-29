"""Preprocess IRFL CSVs and CLIP embeddings for RePercENT.

This script is the non-notebook version of ``src/utils/irfl_preprocess.ipynb``.
It keeps the reproducible pipeline and drops exploratory plots and zero-shot
diagnostics.

Expected local image layout:
    data/irfl/images/<image_id>.jpeg

Outputs:
    data/irfl/datasets/IRFL_train_tensors_2.pt
    data/irfl/datasets/IRFL_test_tensors_2.pt
    data/irfl/datasets/IRFL_train_tensors_aug_2.pt
    data/irfl/datasets/IRFL_test_tensors_aug_2.pt
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("Could not locate the RePercENT repository root")


ROOT_DIR = find_repo_root(Path(__file__).resolve())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.irfl_augmentations import (  # noqa: E402
    DefinitionAugConfig,
    ImageAugConfig,
    make_definition_augmentation_function,
    make_image_augmentation_function,
    make_text_augmentation_function,
)


IRFL_DATASET = "lampent/IRFL"

BASE_DATASETS = {
    "idioms-dataset": "IRFL_idioms_dataset.csv",
    "metaphors-dataset": "IRFL_metaphors_dataset.csv",
    "similes-dataset": "IRFL_similes_dataset.csv",
}

TASK_DATASETS = {
    "idiom-detection-task": "IRFL_idioms_dataset_detect_task.csv",
    "idiom-retrieval-task": "IRFL_idioms_dataset_retrieve_task.csv",
    "metaphor-detection-task": "IRFL_metaphors_dataset_detect_task.csv",
    "metaphor-retrieval-task": "IRFL_metaphors_dataset_retrieve_task.csv",
    "simile-detection-task": "IRFL_similes_dataset_detect_task.csv",
    "simile-retrieval-task": "IRFL_similes_dataset_retrieve_task.csv",
}


ADDITIONAL_DEFINITIONS = {
    "as sharp as an arrow": [
        "Extremely sharp",
        "Keenly perceptive",
        "Quick-witted",
    ],
    "as deep as the ocean": [
        "Extremely deep",
        "Profound in meaning",
        "Intense or complex emotionally",
    ],
    "as white as a ghost": [
        "Extremely pale",
        "Pale from fear or shock",
        "Lacking normal color",
    ],
    "as fresh as a daisy": [
        "Fresh and energetic",
        "Newly rested",
        "Bright and healthy-looking",
    ],
    "as sweet as honey": [
        "Very sweet in taste",
        "Kind and gentle in manner",
        "Pleasant and agreeable",
    ],
    "as bright as the sun": [
        "Extremely bright",
        "Radiant in appearance",
        "Very intelligent",
    ],
    "as solid as a rock": [
        "Very firm and stable",
        "Strongly built",
        "Dependable and reliable",
    ],
    "as helpless as a baby": [
        "Completely helpless",
        "Unable to manage alone",
        "Needing constant assistance",
    ],
    "as sweet as sugar": [
        "Very sweet in taste",
        "Sweet-natured",
        "Kind and pleasant",
    ],
    "as thin as a rail": [
        "Extremely thin",
        "Very slender",
        "Lacking body fat",
    ],
    "as tall as a skyscraper": [
        "Extremely tall",
        "Towering in height",
        "Much taller than average",
    ],
    "as hard as rocks": [
        "Very hard",
        "Tough and unyielding",
        "Difficult to break or change",
    ],
    "as hard as a brick": [
        "Extremely hard",
        "Rigid and unyielding",
        "Difficult to penetrate or affect",
    ],
    "as hot as lava": [
        "Extremely hot",
        "Burning with heat",
        "Intensely heated",
    ],
    "as dry as a desert": [
        "Extremely dry",
        "Lacking moisture",
        "Very arid",
    ],
    "as angry as a hornet": [
        "Extremely angry",
        "Quick-tempered and aggressive",
        "Irritable and ready to attack",
    ],
    "as blue as the ocean": [
        "Deep blue in color",
        "Vividly blue",
        "Very sad or melancholy",
    ],
    "as thin as a stick": [
        "Very thin",
        "Extremely slender",
        "Spare in build",
    ],
    "as sharp as a knife": [
        "Very sharp",
        "Mentally keen",
        "Cutting or biting in speech",
    ],
    "as slow as a turtle": [
        "Very slow",
        "Moving at a sluggish pace",
        "Taking a long time to progress",
    ],
    "as funny as a clown": [
        "Very funny",
        "Comical in behavior",
        "Entertaining and amusing",
    ],
    "as big as a mountain": [
        "Extremely large",
        "Massive in size",
        "Enormous in extent",
    ],
    "as pretty as a princess": [
        "Very pretty",
        "Beautiful in appearance",
        "Charming and attractive",
    ],
    "as tall as a mountain": [
        "Extremely tall",
        "Towering in height",
        "Impressively high",
    ],
    "as slippery as an ice rink": [
        "Very slippery",
        "Hard to hold or keep steady",
        "Difficult to grasp or pin down",
    ],
    "a lion on the battlefield": [
        "A fearless and aggressive fighter",
        "A person showing exceptional courage in combat",
        "Someone who dominates in conflict",
    ],
    "a mighty lion": [
        "A symbol of great strength",
        "A powerful and dominant individual",
        "Someone commanding respect and authority",
    ],
    "a night owl": [
        "A person who stays awake late at night",
        "Someone most active during nighttime",
        "An individual who prefers late hours",
    ],
    "a shinning star": [
        "Someone who stands out prominently",
        "A person admired for talent or success",
        "An outstanding or rising individual",
    ],
    "an angel": [
        "A very kind or helpful person",
        "Someone who shows great compassion",
        "A person regarded as morally pure",
    ],
    "blanket of bullets": [
        "Heavy and continuous gunfire",
        "An overwhelming barrage of shots",
        "Sustained weapons fire over an area",
    ],
    "blanket of flowers": [
        "A surface fully covered with flowers",
        "A dense spread of blossoms",
        "An abundant floral covering",
    ],
    "blanket of snow": [
        "A thick layer of snow",
        "Snow covering the ground completely",
        "A widespread snowfall",
    ],
    "cute teddy bear": [
        "A person who appears gentle and lovable",
        "Someone with a soft or comforting nature",
        "A harmless and friendly individual",
    ],
    "eyes were fireflies": [
        "Eyes shining brightly",
        "Eyes sparkling with excitement",
        "Eyes glowing in the dark",
    ],
    "he drives a tank": [
        "He drives very aggressively",
        "He operates with great force or power",
        "He is unstoppable in movement",
    ],
    "he is a cheetah": [
        "He is extremely fast",
        "He moves with great speed",
        "He acts quickly and efficiently",
    ],
    "he is a fox": [
        "He is clever and cunning",
        "He is skilled in deception",
        "He is attractive or charming",
    ],
    "he was a chicken": [
        "He was cowardly",
        "He lacked courage",
        "He avoided taking risks",
    ],
    "he was a tiger": [
        "He was fierce and aggressive",
        "He showed great strength",
        "He acted with intensity",
    ],
    "heart of a lion": [
        "Great courage",
        "Fearless determination",
        "Bravery in adversity",
    ],
    "heart of gold": [
        "Exceptional kindness",
        "A generous nature",
        "Moral goodness",
    ],
    "heart of stone": [
        "Lack of empathy",
        "Emotional coldness",
        "Inability to feel compassion",
    ],
    "heart sank": [
        "A sudden feeling of disappointment",
        "An immediate loss of hope",
        "A feeling of dread or sadness",
    ],
    "home was prison": [
        "A place of restriction",
        "A situation lacking freedom",
        "An oppressive living environment",
    ],
    "homework is a breeze": [
        "Homework is very easy",
        "Homework requires little effort",
        "Homework can be done quickly",
    ],
    "house of cards": [
        "A fragile system",
        "Something easily destroyed",
        "An unstable structure or plan",
    ],
    "jungle city": [
        "A chaotic urban environment",
        "A dangerous city",
        "A place governed by survival instincts",
    ],
    "light of my life": [
        "A source of happiness",
        "Someone deeply loved",
        "A reason for living",
    ],
    "sea of bees": [
        "A very large number of bees",
        "A swarming mass of insects",
        "An overwhelming presence of bees",
    ],
    "sea of knowledge": [
        "A vast amount of knowledge",
        "Extensive learning or information",
        "Great intellectual depth",
    ],
    "sea of umbrellas": [
        "A large crowd holding umbrellas",
        "An area filled with umbrellas",
        "A dense visual mass of umbrellas",
    ],
    "she is a ray of sunshine": [
        "She brings happiness to others",
        "She has a cheerful personality",
        "She brightens situations",
    ],
    "she is a snake": [
        "She is deceitful",
        "She behaves treacherously",
        "She cannot be trusted",
    ],
    "she was a busy bee": [
        "She was very busy",
        "She was constantly working",
        "She was highly active",
    ],
    "she was a sly cat": [
        "She was clever and sneaky",
        "She acted with quiet cunning",
        "She used subtle deception",
    ],
    "the car is a rocket": [
        "The car is very fast",
        "The car accelerates quickly",
        "The car moves at high speed",
    ],
    "their relationship is a house on fire": [
        "The relationship is intense",
        "The relationship is full of passion",
        "The relationship is unstable or volatile",
    ],
    "walking encyclopedia": [
        "A person with vast knowledge",
        "Someone who knows many facts",
        "An extremely well-informed individual",
    ],
    "wheels of justice": [
        "The legal system in action",
        "The slow process of justice",
        "The operation of law and authority",
    ],
    "as pale as a ghost": [
        "Extremely pale in appearance",
        "Pale from fear or shock",
        "Lacking normal color due to illness",
    ],
    "as sour as vinegar": [
        "Very sour in taste",
        "Unpleasant or bitter in manner",
        "Harsh or disagreeable in tone",
    ],
    "as red as a cherry": [
        "Bright red in color",
        "Vividly colored",
        "Intensely flushed",
    ],
    "as red as a tomato": [
        "Very red in the face",
        "Flushed from embarrassment",
        "Red from heat or exertion",
    ],
    "as happy as a kid": [
        "Very happy",
        "Joyful and carefree",
        "Showing childlike delight",
    ],
    "as slow as a snail": [
        "Extremely slow",
        "Moving with little speed",
        "Progressing at a sluggish pace",
    ],
    "as cold as ice": [
        "Extremely cold",
        "Emotionally distant",
        "Lacking warmth or compassion",
    ],
    "as proud as a peacock": [
        "Very proud",
        "Excessively self-satisfied",
        "Inclined to show off",
    ],
    "as busy as a bee": [
        "Very busy",
        "Constantly active",
        "Engaged in many tasks",
    ],
    "as brave as a lion": [
        "Very brave",
        "Showing great courage",
        "Fearless in danger",
    ],
    "as strong as an ox": [
        "Extremely strong",
        "Physically powerful",
        "Capable of heavy labor",
    ],
    "as smooth as silk": [
        "Very smooth to the touch",
        "Free of roughness",
        "Flowing easily or gracefully",
    ],
    "as sharp as a razor": [
        "Extremely sharp",
        "Quick-witted",
        "Mentally keen",
    ],
    "as sly as a fox": [
        "Clever and deceitful",
        "Skillful in trickery",
        "Cunning in behavior",
    ],
    "as fast as a cheetah": [
        "Extremely fast",
        "Moving at great speed",
        "Capable of rapid action",
    ],
    "as filthy as a pigsty": [
        "Extremely dirty",
        "Covered in filth",
        "Untidy to an extreme degree",
    ],
    "as wild as an animal": [
        "Uncontrolled in behavior",
        "Untamed or savage",
        "Acting without restraint",
    ],
    "as wide as the sea": [
        "Extremely wide",
        "Vast in extent",
        "Seemingly endless",
    ],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_jsonable(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def parse_image_id(value: Any) -> str:
    parsed = as_jsonable(value)
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError(f"Empty image id list: {value!r}")
        parsed = parsed[0]
    return str(parsed).split(".")[0]


def parse_image_list(value: Any) -> list[str]:
    parsed = as_jsonable(value)
    if isinstance(parsed, list):
        return [parse_image_id(v) for v in parsed]
    return [parse_image_id(parsed)]


def parse_definition(cell: Any) -> list[str]:
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    if cell is None or pd.isna(cell):
        return []

    text = str(cell).strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None

    if isinstance(obj, list):
        return [str(x).strip() for x in obj if str(x).strip()]
    if isinstance(obj, str):
        text = obj.strip()

    if text.startswith("[") and text.endswith("]"):
        try:
            obj = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            obj = None
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]

    return [text]


def read_hf_config(config_name: str, split: str) -> pd.DataFrame:
    dataset = load_dataset(IRFL_DATASET, config_name, split=split)
    return pd.DataFrame(dataset)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def build_csvs(datasets_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_frames = {}
    for config_name, filename in BASE_DATASETS.items():
        path = datasets_dir / filename
        df = read_hf_config(config_name, split="dataset")
        save_csv(df, path)
        base_frames[config_name] = df

    task_frames = {}
    for config_name, filename in TASK_DATASETS.items():
        path = datasets_dir / filename
        df = read_hf_config(config_name, split="test")
        save_csv(df, path)
        task_frames[config_name] = df

    complete = pd.concat(
        [
            base_frames["idioms-dataset"],
            base_frames["metaphors-dataset"],
            base_frames["similes-dataset"],
        ],
        ignore_index=True,
    )
    complete_detection = pd.concat(
        [
            task_frames["idiom-detection-task"],
            task_frames["metaphor-detection-task"],
            task_frames["simile-detection-task"],
        ],
        ignore_index=True,
    )
    complete_retrieval = pd.concat(
        [
            task_frames["idiom-retrieval-task"],
            task_frames["metaphor-retrieval-task"],
            task_frames["simile-retrieval-task"],
        ],
        ignore_index=True,
    )

    save_csv(complete, datasets_dir / "IRFL_complete_datasets.csv")
    save_csv(complete_detection, datasets_dir / "IRFL_complete_detection_task_datasets.csv")
    save_csv(complete_retrieval, datasets_dir / "IRFL_complete_retrieval_task_datasets.csv")
    return complete, complete_detection, complete_retrieval


def add_manual_definitions(df: pd.DataFrame, datasets_dir: Path) -> pd.DataFrame:
    df = df.copy()
    if "definition" not in df.columns:
        df["definition"] = pd.NA

    additional_defs_path = datasets_dir / "simile_additional_definitions.json"
    additional_defs_path.write_text(
        json.dumps(
            [
                {"phrase": phrase, "definitions": definitions}
                for phrase, definitions in ADDITIONAL_DEFINITIONS.items()
            ],
            indent=2,
        )
    )

    for phrase, definitions in ADDITIONAL_DEFINITIONS.items():
        mask = df["phrase"] == phrase
        if mask.any():
            df.loc[mask, "definition"] = json.dumps(definitions)
    return df


def fill_missing_task_definitions(task_df: pd.DataFrame, complete_df: pd.DataFrame) -> pd.DataFrame:
    task_df = task_df.copy()
    if "definition" not in task_df.columns:
        task_df["definition"] = pd.NA

    complete_defs = (
        complete_df[["phrase", "definition"]]
        .dropna(subset=["definition"])
        .drop_duplicates(subset=["phrase"])
    )
    definition_by_phrase = dict(zip(complete_defs["phrase"], complete_defs["definition"]))

    missing_mask = task_df["definition"].isna()
    for idx, phrase in task_df.loc[missing_mask, "phrase"].items():
        definition = definition_by_phrase.get(phrase)
        if definition is None:
            for candidate, candidate_definition in definition_by_phrase.items():
                if str(candidate) in str(phrase):
                    definition = candidate_definition
                    break
        if definition is not None:
            task_df.at[idx, "definition"] = definition
    return task_df


def prepare_dataframes(datasets_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete, complete_detection, complete_retrieval = build_csvs(datasets_dir)

    complete = add_manual_definitions(complete, datasets_dir)
    complete_figurative = complete[complete["category"] == "Figurative"].copy()
    save_csv(complete_figurative, datasets_dir / "IRFL_complete_datasets_w_all_defs.csv")
    save_csv(complete, datasets_dir / "IRFL_complete_datasets_full_w_all_defs.csv")

    complete_detection = fill_missing_task_definitions(complete_detection, complete)
    save_csv(
        complete_detection,
        datasets_dir / "IRFL_complete_detection_tasks_w_all_defs.csv",
    )

    complete_retrieval = fill_missing_task_definitions(complete_retrieval, complete)
    save_csv(
        complete_retrieval,
        datasets_dir / "IRFL_complete_retrieval_tasks_w_all_defs.csv",
    )

    train_df, test_df = make_train_test_split(complete_figurative, complete_detection)
    save_csv(train_df, datasets_dir / "IRFL_train_dataset_2.csv")
    save_csv(test_df, datasets_dir / "IRFL_test_detect_dataset_2.csv")
    return train_df, test_df


def make_train_test_split(
    complete_df: pd.DataFrame, detection_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete_phrases = complete_df["phrase"].dropna().unique()
    detection_phrases = detection_df["phrase"].dropna().unique()
    train_phrases = sorted(set(complete_phrases) - set(detection_phrases))

    train_df = complete_df[complete_df["phrase"].isin(train_phrases)].copy()
    test_df = detection_df.copy()

    train_uuids = set(train_df["uuid"].map(parse_image_id))
    test_uuids = set(test_df["answer"].map(lambda value: parse_image_list(value)[0]))
    common_uuids = train_uuids.intersection(test_uuids)
    if common_uuids:
        train_df = train_df[
            ~train_df["uuid"].map(lambda value: parse_image_id(value) in common_uuids)
        ].copy()

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Removed train/test image overlap: {len(common_uuids)} uuids")
    return train_df, test_df


class OpenCLIPTokenExtractor:
    def __init__(self, model_name: str, pretrained: str, device: str) -> None:
        try:
            import open_clip
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "IRFL preprocessing requires open-clip-torch. Install the project "
                "requirements or run inside the Docker image."
            ) from exc

        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def text_tokens(self, texts: str | list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(texts).to(self.device)
        x = self.model.token_embedding(tokens).to(self.device)
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = self.model.ln_final(x)
        mask = tokens != 0
        return x.float(), tokens, mask

    @torch.no_grad()
    def image_patches(self, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        if not all(isinstance(image, Image.Image) for image in images):
            raise ValueError("image_patches expects a list of PIL images")

        visual = self.model.visual
        x = torch.stack([self.preprocess(image.convert("RGB")) for image in images], dim=0)
        x = x.to(device=self.device, dtype=visual.conv1.weight.dtype)

        x = visual.conv1(x)
        batch_size, channels, grid_h, grid_w = x.shape
        x = x.reshape(batch_size, channels, grid_h * grid_w).permute(0, 2, 1)

        cls = visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(
            batch_size, 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = visual.transformer(x)
        x = visual.ln_post(x)

        return x[:, 1:, :].float(), x[:, 0, :].float()


def get_image(image_id: Any, image_dir: Path) -> Image.Image:
    path = image_dir / f"{parse_image_id(image_id)}.jpeg"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing IRFL image: {path}. Place IRFL JPEGs under {image_dir}."
        )
    return Image.open(path).convert("RGB")


def download_irfl_images(image_dir: Path) -> None:
    """Download and flatten the IRFL image archive into image_dir."""
    image_dir.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(
        repo_id=IRFL_DATASET,
        filename="IRFL_images.zip",
        repo_type="dataset",
        local_dir=ROOT_DIR / "data" / "irfl",
    )

    extracted = 0
    with ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            if member.is_dir() or member_path.suffix.lower() not in {".jpeg", ".jpg"}:
                continue

            image_id = parse_image_id(member_path.stem)
            output_path = image_dir / f"{image_id}.jpeg"
            with zf.open(member) as source, output_path.open("wb") as destination:
                destination.write(source.read())
            extracted += 1

    if extracted == 0:
        raise RuntimeError(f"No JPEG images found in downloaded archive: {zip_path}")
    print(f"Extracted {extracted} IRFL JPEGs to: {image_dir}")


def definition_text(row: Any) -> str:
    definitions = parse_definition(row.definition)
    return ". ".join(definitions) + "."


def concat_tensors(items: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(items, dim=0)


def concat_augmented(items: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([item.unsqueeze(0) for item in items], dim=0)

def visualize_test_sample(data_dir: str , image_dir: str, idx: int = None) -> None:
    # load detection task dataframe
    complete_detection_dataset_path = os.path.join(data_dir, "IRFL_complete_detection_tasks_w_all_defs.csv")
    complete_detection_task_datasets = pd.read_csv(complete_detection_dataset_path)
    if idx is None:
        idx = random.randint(0, len(complete_detection_task_datasets) - 1)
    row = complete_detection_task_datasets.iloc[idx]
    distractors = json.loads(row['distractors'])
    answer = json.loads(row['answer'])[0]
    phrase = row['phrase']
    definition = row['definition']

    fig, axes = plt.subplots(1, len(distractors) + 1, figsize=(15, 5))
    for i, img_id in enumerate(distractors):
        img = get_image(img_id, image_dir)
        axes[i].imshow(img)
        axes[i].set_title(f"Distractor {i+1}")
        axes[i].axis('off')

    img = get_image(answer, image_dir)
    axes[-1].imshow(img)
    axes[-1].set_title("Answer")
    axes[-1].axis('off')

    plt.suptitle(f"Phrase: {phrase}\nDefinition: {definition}")
    fig.savefig(ROOT_DIR / "data" / "irfl" / "datasets" / f"test_sample_{idx}.png")
    plt.show()

def extract_base_tensors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_dir: Path,
    datasets_dir: Path,
    extractor: OpenCLIPTokenExtractor,
    output_suffix: str,
) -> None:
    train_images = []
    train_images_in = []
    train_phrases, train_phrases_mask = [], []
    train_phrases_in = []
    train_definitions, train_definitions_mask = [], []
    train_definitions_in = []
    train_joint_def_phrases, train_joint_def_phrases_mask = [], []

    for row in tqdm(train_df.itertuples(), total=len(train_df), desc="Base train"):
        phrase = row.phrase
        definition = definition_text(row)
        joint_def_phrase = f"{phrase}. {definition}"

        image_embeds, _ = extractor.image_patches([get_image(row.uuid, image_dir)])
        phrase_embeds, _, phrase_mask = extractor.text_tokens(phrase)
        definition_embeds, _, definition_mask = extractor.text_tokens(definition)
        joint_embeds, _, joint_mask = extractor.text_tokens(joint_def_phrase)

        train_images.append(image_embeds.cpu())
        train_phrases.append(phrase_embeds.cpu())
        train_phrases_mask.append(phrase_mask.cpu())
        train_definitions.append(definition_embeds.cpu())
        train_definitions_mask.append(definition_mask.cpu())
        train_joint_def_phrases.append(joint_embeds.cpu())
        train_joint_def_phrases_mask.append(joint_mask.cpu())
        train_images_in.append(parse_image_id(row.uuid))
        train_phrases_in.append(phrase)
        train_definitions_in.append(definition)

    test_answers, test_distractors = [], []
    test_images_in, test_images_distractors_in = [], []
    test_phrases, test_phrases_mask = [], []
    test_phrases_in = []
    test_definitions, test_definitions_mask = [], []
    test_definitions_in = []
    test_joint_def_phrases, test_joint_def_phrases_mask = [], []
    test_figurative_type = []

    for row in tqdm(test_df.itertuples(), total=len(test_df), desc="Base test"):
        distractors = parse_image_list(row.distractors)
        answer = parse_image_list(row.answer)[0]
        phrase = row.phrase
        definition = definition_text(row)
        joint_def_phrase = f"{phrase}. {definition}"

        distractor_embeds, _ = extractor.image_patches(
            [get_image(image_id, image_dir) for image_id in distractors]
        )
        answer_embeds, _ = extractor.image_patches([get_image(answer, image_dir)])
        phrase_embeds, _, phrase_mask = extractor.text_tokens(phrase)
        definition_embeds, _, definition_mask = extractor.text_tokens(definition)
        joint_embeds, _, joint_mask = extractor.text_tokens(joint_def_phrase)

        test_distractors.append(distractor_embeds.cpu())
        test_answers.append(answer_embeds.cpu())
        test_phrases.append(phrase_embeds.cpu())
        test_phrases_mask.append(phrase_mask.cpu())
        test_definitions.append(definition_embeds.cpu())
        test_definitions_mask.append(definition_mask.cpu())
        test_joint_def_phrases.append(joint_embeds.cpu())
        test_joint_def_phrases_mask.append(joint_mask.cpu())
        test_figurative_type.append(row.figurative_type)
        test_images_distractors_in.append(distractors)
        test_images_in.append(answer)
        test_phrases_in.append(phrase)
        test_definitions_in.append(definition)

    torch.save(
        {
            "train_images": concat_tensors(train_images),
            "train_images_in": train_images_in,
            "train_phrases": concat_tensors(train_phrases),
            "train_phrases_in": train_phrases_in,
            "train_phrases_mask": concat_tensors(train_phrases_mask),
            "train_definitions": concat_tensors(train_definitions),
            "train_definitions_mask": concat_tensors(train_definitions_mask),
            "train_definitions_in": train_definitions_in,
            "train_joint_def_phrases": concat_tensors(train_joint_def_phrases),
            "train_joint_def_phrases_mask": concat_tensors(train_joint_def_phrases_mask),
        },
        datasets_dir / f"IRFL_train_tensors{output_suffix}.pt",
    )

    torch.save(
        {
            "test_answers": concat_tensors(test_answers),
            "test_images_in": test_images_in,
            "test_distractors": concat_augmented(test_distractors),
            "test_images_distractors_in": test_images_distractors_in,
            "test_phrases": concat_tensors(test_phrases),
            "test_phrases_in": test_phrases_in,
            "test_phrases_mask": concat_tensors(test_phrases_mask),
            "test_definitions": concat_tensors(test_definitions),
            "test_definitions_in": test_definitions_in,
            "test_definitions_mask": concat_tensors(test_definitions_mask),
            "test_joint_def_phrases": concat_tensors(test_joint_def_phrases),
            "test_joint_def_phrases_mask": concat_tensors(test_joint_def_phrases_mask),
            "test_figurative_type": test_figurative_type,
        },
        datasets_dir / f"IRFL_test_tensors{output_suffix}.pt",
    )


def extract_augmented_tensors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_dir: Path,
    datasets_dir: Path,
    extractor: OpenCLIPTokenExtractor,
    output_suffix: str,
    seed: int,
) -> None:
    image_augment = make_image_augmentation_function(
        ImageAugConfig(), seed=seed
    )
    phrase_augment = make_text_augmentation_function()
    definition_augment = make_definition_augmentation_function(
        DefinitionAugConfig(seed=seed)
    )

    train_images_aug = []
    train_phrases_aug, train_phrases_mask_aug = [], []
    train_definitions_aug, train_definitions_mask_aug = [], []
    train_joint_def_phrases_aug, train_joint_def_phrases_mask_aug = [], []

    for row in tqdm(train_df.itertuples(), total=len(train_df), desc="Aug train"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        phrase = row.phrase
        image = get_image(row.uuid, image_dir)
        definition = definition_text(row)

        augmented_images = image_augment(image, augment_types=2)
        augmented_phrases = phrase_augment(phrase)
        augmented_definitions = definition_augment(definition)
        augmented_joint = [
            f"{aug_phrase}. {aug_definition}."
            for aug_phrase in augmented_phrases
            for aug_definition in augmented_definitions
        ]

        image_embeds, _ = extractor.image_patches(augmented_images)
        phrase_embeds, _, phrase_mask = extractor.text_tokens(augmented_phrases)
        definition_embeds, _, definition_mask = extractor.text_tokens(
            augmented_definitions
        )
        joint_embeds, _, joint_mask = extractor.text_tokens(augmented_joint)

        train_images_aug.append(image_embeds.cpu())
        train_phrases_aug.append(phrase_embeds.cpu())
        train_phrases_mask_aug.append(phrase_mask.cpu())
        train_definitions_aug.append(definition_embeds.cpu())
        train_definitions_mask_aug.append(definition_mask.cpu())
        train_joint_def_phrases_aug.append(joint_embeds.cpu())
        train_joint_def_phrases_mask_aug.append(joint_mask.cpu())

    test_answers_aug = []
    test_phrases_aug, test_phrases_mask_aug = [], []
    test_definitions_aug, test_definitions_mask_aug = [], []
    test_joint_def_phrases_aug, test_joint_def_phrases_mask_aug = [], []

    for row in tqdm(test_df.itertuples(), total=len(test_df), desc="Aug test"):
        answer = parse_image_list(row.answer)[0]
        phrase = row.phrase
        definition = definition_text(row)

        augmented_answer = image_augment(get_image(answer, image_dir), augment_types=2)
        augmented_phrases = phrase_augment(phrase)
        augmented_definitions = definition_augment(definition)
        augmented_joint = [
            f"{aug_phrase}. {aug_definition}."
            for aug_phrase in augmented_phrases
            for aug_definition in augmented_definitions
        ]

        answer_embeds, _ = extractor.image_patches(augmented_answer)
        phrase_embeds, _, phrase_mask = extractor.text_tokens(augmented_phrases)
        definition_embeds, _, definition_mask = extractor.text_tokens(
            augmented_definitions
        )
        joint_embeds, _, joint_mask = extractor.text_tokens(augmented_joint)

        test_answers_aug.append(answer_embeds.cpu())
        test_phrases_aug.append(phrase_embeds.cpu())
        test_phrases_mask_aug.append(phrase_mask.cpu())
        test_definitions_aug.append(definition_embeds.cpu())
        test_definitions_mask_aug.append(definition_mask.cpu())
        test_joint_def_phrases_aug.append(joint_embeds.cpu())
        test_joint_def_phrases_mask_aug.append(joint_mask.cpu())

    torch.save(
        {
            "train_images_aug": concat_augmented(train_images_aug).half(),
            "train_phrases_aug": concat_augmented(train_phrases_aug).half(),
            "train_phrases_mask_aug": concat_augmented(train_phrases_mask_aug).half(),
            "train_definitions_aug": [item.half() for item in train_definitions_aug],
            "train_definitions_mask_aug": [
                item.half() for item in train_definitions_mask_aug
            ],
            "train_joint_def_phrases_aug": [
                item.half() for item in train_joint_def_phrases_aug
            ],
            "train_joint_def_phrases_mask_aug": [
                item.half() for item in train_joint_def_phrases_mask_aug
            ],
        },
        datasets_dir / f"IRFL_train_tensors_aug{output_suffix}.pt",
    )

    torch.save(
        {
            "test_answers_aug": concat_augmented(test_answers_aug).half(),
            "test_phrases_aug": concat_augmented(test_phrases_aug).half(),
            "test_phrases_mask_aug": concat_augmented(test_phrases_mask_aug).half(),
            "test_definitions_aug": [item.half() for item in test_definitions_aug],
            "test_definitions_mask_aug": [
                item.half() for item in test_definitions_mask_aug
            ],
            "test_joint_def_phrases_aug": [
                item.half() for item in test_joint_def_phrases_aug
            ],
            "test_joint_def_phrases_mask_aug": [
                item.half() for item in test_joint_def_phrases_mask_aug
            ],
        },
        datasets_dir / f"IRFL_test_tensors_aug{output_suffix}.pt",
    )


def load_or_prepare_dataframes(datasets_dir: Path, force_download: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = datasets_dir / "IRFL_train_dataset_2.csv"
    test_path = datasets_dir / "IRFL_test_detect_dataset_2.csv"
    if train_path.exists() and test_path.exists() and not force_download:
        print("Using existing train/test CSVs.")
        return pd.read_csv(train_path), pd.read_csv(test_path)
    return prepare_dataframes(datasets_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess IRFL data for RePercENT")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=ROOT_DIR / "data" / "irfl" / "datasets",
        help="Directory for generated IRFL CSVs and tensors.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ROOT_DIR / "data" / "irfl" / "images",
        help="Directory containing IRFL JPEG images named <image_id>.jpeg.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for CLIP embedding extraction.",
    )
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument(
        "--output-suffix",
        default="_2",
        help="Suffix for tensor files. Default matches training/main_irfl.py default loaded files.",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Regenerate CSVs from Hugging Face even if train/test CSVs exist.",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only build CSV train/test files; skip CLIP tensor extraction.",
    )
    parser.add_argument(
        "--skip-augmented",
        action="store_true",
        help="Skip augmented tensor extraction.",
    )
    parser.add_argument(
        "--visualize-test-sample",
        action="store_true",
        help="Visualize a test sample with one correct image and three distractor images from the detection task."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.datasets_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_or_prepare_dataframes(
        args.datasets_dir, force_download=args.force_download
    )
    if args.csv_only:
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    has_images = args.image_dir.exists() and any(args.image_dir.glob("*.jpeg"))
    if not has_images:
        print(
            f"IRFL JPEGs were not found in {args.image_dir}. "
            "Downloading and extracting them before extracting tensors."
        )
        download_irfl_images(args.image_dir)

    extractor = OpenCLIPTokenExtractor(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
    )

    if args.visualize_test_sample:
        visualize_test_sample(data_dir=args.datasets_dir, image_dir=args.image_dir)

    
    extract_base_tensors(
        train_df=train_df,
        test_df=test_df,
        image_dir=args.image_dir,
        datasets_dir=args.datasets_dir,
        extractor=extractor,
        output_suffix=args.output_suffix,
    )
    if not args.skip_augmented:
        extract_augmented_tensors(
            train_df=train_df,
            test_df=test_df,
            image_dir=args.image_dir,
            datasets_dir=args.datasets_dir,
            extractor=extractor,
            output_suffix=args.output_suffix,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
