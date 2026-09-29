from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.generation_logits_process import LogitsProcessor, LogitsProcessorList
import torch

from . import config


class RestrictToTextVocab(LogitsProcessor):
    """Bans OFA's non-text vocab (image-codebook / bbox-bin tokens) during captioning."""

    def __init__(self, text_vocab_size):
        self.text_vocab_size = text_vocab_size

    def __call__(self, input_ids, scores):
        scores[:, self.text_vocab_size:] = -float("inf")
        return scores


class BiomedGPTCaptioner:
    def __init__(self, checkpoint_dir=config.CHECKPOINT_DIR, device=config.DEVICE, generation_params=None):
        self.device = device
        self.generation_params = generation_params or config.GENERATION_PARAMS
        self.tokenizer = OFATokenizer.from_pretrained(checkpoint_dir)
        self.model = OFAModel.from_pretrained(checkpoint_dir, use_cache=False)
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize(
                (config.IMAGE_RESOLUTION, config.IMAGE_RESOLUTION),
                interpolation=Image.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
        ])

        prompt_ids = self.tokenizer([config.CAPTION_PROMPT], return_tensors="pt").input_ids
        self.prompt_ids = prompt_ids.to(device)

        self.logits_processor = LogitsProcessorList([
            RestrictToTextVocab(config.TEXT_VOCAB_SIZE)
        ])

    def describe(self, image_path):
        image = Image.open(image_path)
        patch_image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                self.prompt_ids,
                patch_images=patch_image,
                logits_processor=self.logits_processor,
                **self.generation_params,
            )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
