import logging
import dtlpy as dl
from typing import List
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from hugging_base import HuggingBase

logger = logging.getLogger("[PEGASUS]")

class HuggingAdapter(HuggingBase):
    def load_model_and_processor(self):
        self.tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        self.model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    def prepare_item_func(self, item: dl.Item):
        """
        Validates if the item is of supported mimetype.

        Args:
            item: The item to validate

        Returns:
            The validated item
        """
        if "text" not in item.mimetype and "json" not in item.mimetype:
            raise ValueError("Item must be of type 'text' or a JSON prompt item for summarization.")
        return item

    def predict(self, batch: List[dl.Item], **kwargs):
        """
        Generate summaries for the text in the items and update them.

        Args:
            batch: A list of items to process

        Returns:
            List of processed items
        """
        for item in batch:
            try:
                # Extract text based on item type
                if "text" in item.mimetype:
                    text = item.download(save_locally=False).read().decode("utf-8")
                    is_prompt = False
                else:
                    # Handle JSON prompt item
                    prompt_item = dl.PromptItem.from_item(item)
                    messages = prompt_item.to_messages(model_name=self.model_name)
                    text = None
                    for message in messages:
                        content = message["content"][0].get("text", None)
                        if content:
                            text = content
                            break

                    if not text:
                        logger.error(f"No text found in prompt item {item.id}. Skipping...")
                        continue
                    is_prompt = True

                # Generate summary
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
                output = self.model.generate(**inputs, max_length=60, num_beams=5, early_stopping=True)
                summary = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

                # Update the item based on its type
                if is_prompt:
                    # Add a response to the prompt item
                    response = dl.Prompt(key='summary-response')
                    response.add_element(mimetype=dl.PromptType.TEXT, value=summary)
                    prompt_item.prompts.append(response)
                    updated_item = item.dataset.items.upload(prompt_item, overwrite=True)
                else:
                    # Update metadata for text items
                    if 'user' not in item.metadata:
                        item.metadata['user'] = dict()
                    item.metadata['user']['summary'] = summary
                    updated_item = item.update()

                logger.debug(f"Generated summary for item {item.id}: {summary[:100]}...")

            except Exception as e:
                logger.error(f"Error processing item {item.id}: {str(e)}")

        return []

    def get_cmd(self):
        return "python script.py --text <text_input>"

# Example usage
if __name__ == "__main__":
    adapter = HuggingAdapter()
    adapter.load("path/to/model/config")
    print(adapter.get_cmd())