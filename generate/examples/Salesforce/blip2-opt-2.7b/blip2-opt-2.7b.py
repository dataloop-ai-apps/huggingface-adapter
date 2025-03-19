from hugging_base import HuggingBase
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class HuggingAdapter(HuggingBase):
    """
    Implementation of the HuggingBase class for the BLIP-2 model.
    """

    def load_model_and_processor(self):
        """
        Load the BLIP-2 model and processor.
        """
        self.model_name = self.configuration.get("model_name", "blip-2")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    def get_cmd(self):
        return "python -m blip2-opt-2.7b"
