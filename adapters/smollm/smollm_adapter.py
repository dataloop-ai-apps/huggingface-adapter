import dtlpy as dl
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger("[smollm-1-7b]")


class HuggingAdapter(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        logger.info("Loading SmolLM adapter...")
        self.model_path = self.configuration.get("model_path", "HuggingFaceTB/SmolLM2-360M-Instruct")
        self.max_tokens = self.configuration.get('max_tokens', 1024)
        self.temperature = self.configuration.get('temperature', 0.2)
        self.top_p = self.configuration.get('top_p', 0.9)
        self.stream = self.configuration.get('stream', True)
        self.system_prompt = self.configuration.get('system_prompt', None)
        
        logger.info(f"Model configuration - Path: {self.model_path}, Max tokens: {self.max_tokens}, Temperature: {self.temperature}, Top-p: {self.top_p}, Stream: {self.stream}")
        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        
        try:
            logger.info(f"Loading tokenizer from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            logger.info("Tokenizer loaded successfully")
            
            logger.info(f"Loading model from {self.model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise

    def prepare_item_func(self, item: dl.Item):
        logger.debug(f"Preparing item: {item.id}")
        prompt_item = dl.PromptItem.from_item(item=item)
        logger.debug(f"Created prompt item with {len(prompt_item.prompts)} prompts")
        return prompt_item

    def train(self, data_path, output_path, **kwargs):
        logger.warning("Training method called but not implemented")
        raise NotImplementedError("Training not implemented yet")

    def _reformat_messages(self, messages):
        """Convert messages to format expected by the model."""
        logger.debug(f"Reformatting {len(messages)} messages")
        formatted_messages = []
        for i, msg in enumerate(messages):
            role = msg.get('role', 'user')
            content = msg.get('content', [])
            
            # Extract text content from the message
            text_content = ""
            if isinstance(content, str):
                text_content = content
                logger.debug(f"Message {i}: {role} - string content ({len(text_content)} chars)")
            elif isinstance(content, list):
                for element in content:
                    if isinstance(element, dict) and element.get('type') == 'text':
                        text_content += element.get('text', '')
                    elif isinstance(element, dict) and element.get('mimetype') == dl.PromptType.TEXT:
                        text_content += element.get('value', '')
                logger.debug(f"Message {i}: {role} - list content ({len(text_content)} chars)")
            
            if text_content:
                formatted_messages.append({
                    'role': role,
                    'content': text_content
                })
            else:
                logger.warning(f"Message {i} has no text content")
        
        logger.debug(f"Reformatted to {len(formatted_messages)} messages")
        return formatted_messages
    
    def _get_response(self, messages):
        """Generate response from the model."""
        logger.debug("Generating response from model")
        try:
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            logger.debug(f"Applied chat template, input length: {len(input_text)} chars")
            
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            logger.debug(f"Tokenized input, tensor shape: {inputs.shape}")
            
            logger.info(f"Starting generation with max_tokens={self.max_tokens}, temperature={self.temperature}, top_p={self.top_p}")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True
                )
            
            output_text = self.tokenizer.decode(outputs[0])
            logger.debug(f"Generated output length: {len(output_text)} chars")
            
            lines = output_text.split('<|im_end|>')
            last_assistant_response = None
            
            # Extract the last assistant response
            for line in lines:
                if '<|im_start|>assistant' in line:
                    last_assistant_response = line.split('<|im_start|>assistant\n')[-1].strip()
            
            if last_assistant_response:
                logger.info(f"Extracted assistant response ({len(last_assistant_response)} chars)")
            else:
                logger.warning("Could not extract assistant response from generated text")
            
            return last_assistant_response or ""
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            raise
    
    def _stream_response(self, messages):
        """Generate streaming response from the model."""
        logger.debug("Starting streaming response")
        response = self._get_response(messages)
        
        # Simulate streaming by yielding chunks
        chunk_size = 10  # Characters per chunk
        total_chunks = (len(response) + chunk_size - 1) // chunk_size
        logger.debug(f"Streaming response in {total_chunks} chunks of {chunk_size} chars each")
        
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            logger.debug(f"Yielding chunk {i//chunk_size + 1}/{total_chunks}")
            yield chunk
    
    def predict(self, batch, **kwargs):
        logger.info(f"Starting prediction for batch of {len(batch)} items")
        
        for idx, prompt_item in enumerate(batch):
            logger.info(f"Processing item {idx + 1}/{len(batch)}")
            prompt_item: dl.PromptItem = prompt_item
            
            _messages = prompt_item.to_messages(model_name=self.model_entity.name)
            logger.debug(f"Converted to {len(_messages)} messages")
            
            messages = self._reformat_messages(_messages)
            
            if self.system_prompt is not None:
                logger.debug("Adding system prompt to messages")
                messages.insert(0, {"role": "system", "content": self.system_prompt})
                
            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', []) if prompt_item.prompts else []
            if len(nearest_items) > 0:
                logger.info(f"Found {len(nearest_items)} nearest items, building context")
                context = prompt_item.build_context(nearest_items=nearest_items,
                                                    add_metadata=self.configuration.get("add_metadata"))
                logger.info(f"Nearest items Context: {context}")
                messages.append({"role": "assistant", "content": context})

            if self.stream:
                logger.info("Using streaming mode")
                stream_response = self._stream_response(messages=messages)
                response = ""
                chunk_count = 0
                for chunk in stream_response:
                    response += chunk
                    chunk_count += 1
                logger.info(f"Received {chunk_count} chunks, total response length: {len(response)} chars")
                
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})
            else:
                logger.info("Using non-streaming mode")
                response = self._get_response(messages=messages)
                logger.info(f"Generated response length: {len(response)} chars")
                
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})
            
            logger.debug(f"Completed processing item {idx + 1}")
        
        logger.info("Prediction batch completed")
        return []

if __name__ == "__main__":
    import dtlpy as dl
    dl.setenv('prod')
    if dl.token_expired():
        dl.login()
    project: dl.Project = dl.projects.get(project_id='<project_id>')
    dataset: dl.Dataset = dl.datasets.get(dataset_id='<dataset_id>')
    model_entity = project.models.get(model_id="<model_id>")
    item = dl.items.get(item_id="<item_id>")
    adapter = HuggingAdapter(model_entity=model_entity)
    prompt_item = dl.PromptItem.from_item(item=item)
    response = adapter.predict([prompt_item])
    print(response)
