import os
import sys
import json
import importlib.util
import dtlpy as dl


# TODO convert this into a test
def load_class_from_file(subdir, module_name, class_name):
    """Loads a class dynamically from a module in a subdirectory."""
    module_path = os.path.join(os.path.dirname(__file__), subdir, f"{module_name}.py")

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module '{module_name}' not found in '{subdir}'.")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'.")

    return getattr(module, class_name)


# Example Usage
if __name__ == "__main__":
    # adapter_path = "vit-gpt2-image-captioning_2025.03.03_14.19.05"
    # model_json_path = r"C:\Users\Yaya Tang\PycharmProjects\huggingface-adapter\adapters\vit_gpt2_image_captioning\dataloop.json"
    adapter_path_stem = "bert-base-NER_2025.03.06_10.25.17"

    # # load model json to create a model entity
    # model_json_path = r"C:\Users\Yaya Tang\PycharmProjects\huggingface-adapter\adapters\bert_base_ner\dataloop.json"
    # with open(model_json_path, "r") as f:
    #     manifest = json.load(f)
    # model_json = manifest['components']['models'][0]

    model_json = {
        "name": "model-adapter",
        "moduleName": adapter_path_stem,
        "scope": "project",
        "status": "pre-trained",
        "configuration": {"module_name": "adapters.bert_base_ner.bert_base_ner", "device": "cpu"},
        "metadata": {},
    }

    ModelAdapter = load_class_from_file(subdir="responses", module_name=adapter_path_stem, class_name="HuggingAdapter")
    model_entity = dl.Model.from_json(_json=model_json, client_api=dl.client_api, project=None, package=dl.Package())

    model_app = ModelAdapter(model_entity=model_entity)
    print(f"Loaded class: {model_app}")

    dl.setenv("rc")
    project = dl.projects.get(project_name="Model mgmt demo")
    dataset = project.datasets.get(dataset_name="llama_testing")
    item = dataset.items.get(item_id="67c961cc084575d04c468719")
    # pitem = dl.PromptItem.from_item(item=item)
    # for prompt in pitem.prompts:
    #     print(prompt.key)
    #     prompt.key = "1"

    item.annotations.list().delete()
    preds = model_app.predict_items(items=[item])

    for pred in preds:
        print(pred)
        print(pred.annotations.to_json())
