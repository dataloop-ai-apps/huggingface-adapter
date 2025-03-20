# hand edited for mimetypes
import logging
import dtlpy as dl
from typing import List
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

logger = logging.getLogger("[PEGASUS]")

class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        self.model_name = self.configuration.get("model_name", "pegasus")
        self.device = self.configuration.get("device", "cpu")
        self.tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        self.model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
        self.model.to(self.device)

    def prepare_item_func(self, item: dl.Item):
        if "text/" not in item.mimetype or "*/json" not in item.mimetype:
            raise ValueError("Item must be of type 'text' or 'json' for summarization.")
        else:
            if "text/" in item.mimetype:
                text = item.download().read().decode("utf-8")
            else:
                prompt_item = dl.PromptItem.from_item(item) 
                prompts = prompt_item.prompts
                for prompt in prompts:
                    for element in prompts[prompt]:
                        if element['mimetype'] == 'application/text':
                            text = element['value']
                            break
        return text

    def predict(self, batch: List[dl.Item], **kwargs):
        for prompt_item in batch:
            prompt_txt = prompt_item.content.get("value", "")
            if not prompt_txt:
                raise ValueError("No text found in prompt item.")

            inputs = self.tokenizer(prompt_txt, return_tensors="pt", truncation=True, padding=True).to(self.device)
            output = self.model.generate(**inputs, max_length=60, num_beams=5, early_stopping=True)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            logger.debug("Response: {}".format(response))
            prompt_item.add(
                message={
                    "role": "assistant",
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": response}],
                },
                model_info={
                    "name": self.model_name,
                    "confidence": 1.0,
                    "model_id": self.model_entity.id,
                },
            )
        return []
