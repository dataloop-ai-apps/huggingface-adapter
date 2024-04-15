import dtlpy as dl
import torch
import logging
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

logger = logging.getLogger("[ivrit-ai/whisper-large-v2-tuned]")


class HuggingAdapter:
    def __init__(self, configuration):
        model_path = configuration.get("model_path", 'ivrit-ai/whisper-large-v2-tuned')
        self.processor = WhisperProcessor.from_pretrained(model_path)
        torch_dtype = configuration.get("torch_dtype", torch.float32) if torch.cuda.is_available() else torch.float32
        self.device = configuration.get("device", "cpu")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path,
                                                                     torch_dtype=torch_dtype)
        if "cuda" in self.device and not torch.cuda.is_available():
            logger.warning("Configuration set device to CUDA, but no GPU is available. Switching to CPU")
            self.device = "cpu"
        self.model.to(self.device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=configuration.get("max_new_tokens", 128),
            chunk_length_s=configuration.get("chunk_length", 30),
            batch_size=configuration.get("batch_size", 16),
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=self.device
            )
        self.sampling_rate = configuration.get("sampling_rate", 16000)
        self.num_beams = configuration.get("num_beams", 5)

    def prepare_item_func(self, item: dl.Item):
        return item

    def train(self, data_path, output_path, **kwargs):
        logger.info("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for item in batch:
            filename = item.download(overwrite=True)
            result = self.pipe(filename)
            transcript = ''
            builder = item.annotations.builder()
            for chunk in result['chunks']:
                text = chunk['text']
                transcript += text
                timestamp = chunk['timestamp']
                start = timestamp[0]
                end = timestamp[1]
                builder.add(annotation_definition=dl.Subtitle(label="Transcript",text=text),
                            start_time=start,
                            end_time=end,
                            model_info={
                                "name": "ivrit-ai/whisper-large-v2-tuned",
                                "confidence": 1.0
                                })
            annotations.append(builder)
            os.remove(filename)
            logger.debug(f"Transcript: {transcript}")
        return annotations
