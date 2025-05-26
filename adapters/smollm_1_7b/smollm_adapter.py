import dtlpy as dl
import torch
import json
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger("[smollm-1-7b]")


class HuggingAdapter:
    def __init__(self, configuration):
        # Load model and tokenizer
        model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct" # You can also use other variants like smollm-2.7b, etc.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)


    def prepare_item_func(self, item: dl.Item):
        buffer = json.load(item.download(save_locally=False))
        return buffer

    def train(self, data_path, output_path, **kwargs):
        raise NotImplementedError("Training not implemented yet")

    def predict(self, batch, **kwargs):
        annotations = []
        for item in batch:
            prompts = item["prompts"]
            item_annotations = dl.AnnotationCollection()
            for prompt_key, prompt_content in prompts.items():
                questions = list(prompt_content.values()) if isinstance(prompt_content, dict) else prompt_content

                for question in questions:
                    if question["mimetype"] == dl.PromptType.TEXT:
                        logger.info(f"User: {question['value']}")
                        inputs = self.tokenizer(question['value'], return_tensors="pt")
                        with torch.no_grad():
                            outputs = self.model.generate(
                                inputs.input_ids,
                                max_length=100,
                                temperature=0.7,
                                top_p=0.9,
                                do_sample=True
                                )
                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        logger.info("Response: {}".format(generated_text))
                        item_annotations.add(annotation_definition=dl.FreeText(text=generated_text), prompt_id=prompt_key,
                                             model_info={
                                                 "name": "Smollm 1.7b",
                                                 "confidence": 1.0
                        })
                    else:
                        logger.warning(f"Model only accepts text prompts, ignoring the current prompt.")
            annotations.append(item_annotations)
        return annotations
    
    
def main():
    import time
    adapter = HuggingAdapter(configuration=None)
    item = dl.items.get (item_id="65e0586075f05f7af477c045")
    batch = adapter.prepare_item_func(item)
    tic = time.time()
    response = adapter.predict([batch])
    toc = time.time()
    print(response)
    print(f"Time taken: {toc - tic} seconds")

if __name__ == "__main__":
    ...
