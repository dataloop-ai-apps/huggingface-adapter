import os
import sys
import glob
import importlib.util
import dtlpy as dl

MODEL_NAME = "microsoft/kosmos-2-patch14-224"  # "facebook/detr-resnet-50"  # "dslim/bert-base-NER"
ITEM_ID = "6880d943c1294f3c665f6e78"
# Available items:
# "67c961cc084575d04c468719"  -  photo in a prompt item
# "67da8d6da1b8775dbed524f2" - text only prompt item
# "67bee79dcb4162a4b84e1ab5" - image item
# "67dbd95b45c52a84598f6793" - text item
# "67e16db0f27dd1335f907618" - text prompt item for NER
# "6880d943c1294f3c665f6e78" - image item, snowman from HuggingFace microsoft kosmos-2-patch14-224


def find_most_recent_adapter(model_name, responses_dir):
    """Find the most recent adapter script for a given model name.

    Args:
        model_name: Name of the model (e.g. 'bert-base-NER')
        responses_dir: Directory containing the adapter files

    Returns:
        str: Name of the most recent adapter file

    The function looks for files matching the pattern:
    {model_name}_{media_type}_{model_type}_{timestamp}.py
    """
    # Remove any directory prefix from model name
    model_name = model_name.split('/')[-1]

    # Pattern matches: model_name_*_*_*.py
    # This will match the new format: model_name_media_type_model_type_timestamp.py
    pattern = os.path.join(responses_dir, f"{model_name}_*_*_*.py")
    adapter_files = glob.glob(pattern)

    # If no files found in main responses dir, check the basemodeladapter subdirectory
    if not adapter_files:
        base_adapter_dir = os.path.join(responses_dir, "basemodeladapter")
        if os.path.exists(base_adapter_dir):
            pattern = os.path.join(base_adapter_dir, f"{model_name}_*_*_*.py")
            adapter_files = glob.glob(pattern)

    if not adapter_files:
        raise FileNotFoundError(
            f"No adapter files found for model '{model_name}' in responses dir or basemodeladapter subdir"
        )

    # Sort by filename (which includes timestamp) in descending order
    latest_file = max(adapter_files)
    return os.path.basename(latest_file)


def load_class_from_file(subdir, module_name, class_name):
    """Loads a class dynamically from a module in a subdirectory."""
    # Remove any directory prefix from module name and .py extension if present
    module_name = module_name.split('/')[-1].replace('.py', '')
    module_path = os.path.join(os.path.dirname(__file__), subdir, f"{module_name}.py")

    if not os.path.exists(module_path):
        # Check basemodeladapter subdirectory if file not found
        base_adapter_dir = os.path.join(os.path.dirname(__file__), subdir, "basemodeladapter")
        alternate_path = os.path.join(base_adapter_dir, f"{module_name}.py")

        if os.path.exists(alternate_path):
            module_path = alternate_path
        else:
            raise FileNotFoundError(
                f"Module '{module_name}' not found in '{subdir}' or its basemodeladapter subdirectory."
            )

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'.")

    return getattr(module, class_name)


if __name__ == "__main__":
    responses_dir = os.path.join(os.path.dirname(__file__), "responses")
    adapter_filename = find_most_recent_adapter(MODEL_NAME.split('/')[-1], responses_dir)
    print(f"Using most recent adapter: {adapter_filename}")

    # Extract media_type and model_type from filename (model_name_media_type_model_type_timestamp.py)
    media_type, model_type = adapter_filename.split('_')[1:3]
    model_json = {
        "name": "model-adapter",
        "moduleName": adapter_filename,
        "scope": "project",
        "status": "pre-trained",
        "configuration": {
            "module_name": adapter_filename,
            "device": "cpu",
            "model_name": MODEL_NAME,
            "media_type": media_type,
            "model_type": model_type,
        },
        "metadata": {},
    }

    ModelAdapter = load_class_from_file(subdir="responses", module_name=adapter_filename, class_name="HuggingAdapter")
    model_entity = dl.Model.from_json(_json=model_json, client_api=dl.client_api, project=None, package=dl.Package())

    model_app = ModelAdapter(model_entity=model_entity)
    print(f"Loaded class: {model_app}")

    dl.setenv("rc")
    project = dl.projects.get(project_name="Model mgmt demo RC")
    dataset = project.datasets.get(dataset_name="llama_testing")
    item = dataset.items.get(item_id=ITEM_ID)

    item.annotations.list().delete()
    # returns a list of annotations
    items, annotations = model_app.predict_items(items=[item])

    for annotation in annotations:
        print(annotation)

    print("\n\neverything looks good :)\n\n")
