# OWLv2 Object Detection Adapter

## Overview

OWLv2 (Open-World Localization v2) is a zero-shot, text-conditioned object detection model developed by Google Research. Unlike traditional object detectors that require training on specific categories, OWLv2 can detect objects based on natural language descriptions without category-specific training data.

**Key Features:**
- **Zero-shot detection**: Detect objects without prior training on those categories
- **Text-conditioned**: Use natural language to describe what to detect
- **Open-vocabulary**: Works with arbitrary object descriptions
- **Flexible label sources**: Two modes for providing detection queries

## Model Information

- **Provider**: Google Research
- **License**: Apache 2.0
- **Base Model**: `google/owlv2-base-patch16`
- **Paper**: [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)

## How It Works

OWLv2 detects objects in images using text queries. You can configure the source of these text queries using the `label_source` parameter.

### Two Configuration Modes

#### 1. Dataset Labels Mode (Default)
Uses labels from the Dataloop dataset.

```json
{
  "label_source": "dataset_labels"
}
```

**Use case**: Standard workflow when working with a labeled dataset. The model will detect all label categories defined in your dataset.

**Example**: If your dataset has labels `["person", "car", "bicycle"]`, OWLv2 will search for these objects in each image.

#### 2. Custom Labels Mode
Uses a user-defined list of labels from the configuration.

```json
{
  "label_source": "custom",
  "custom_labels": ["cat", "dog", "bird", "fish"]
}
```

**Use case**: Quick experiments with specific object categories without creating dataset labels. Useful for testing or ad-hoc detection tasks.

**Example**: Detect specific objects across a collection of images without labeling the dataset first.

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|

| `model_weights` | string | `"google/owlv2-base-patch16"` | HuggingFace model identifier |
| `confidence_threshold` | float | `0.2` | Minimum confidence score for detections (0.0-1.0) |
| `label_source` | string | `"dataset_labels"` | Label source: `"dataset_labels"` or `"custom"` |
| `custom_labels` | list | `[]` | List of labels when using `label_source: "custom"` |

### Available Model Checkpoints

You can change the `model_checkpoint` to use different OWLv2 variants:

| Checkpoint | Size | Performance | Speed |
|------------|------|-------------|-------|
| `google/owlv2-base-patch16` | Base | Good | Fast |
| `google/owlv2-base-patch16-ensemble` | Base (ensemble) | Better | Medium |
| `google/owlv2-large-patch14` | Large | Best | Slower |

## Usage Examples

### Example 1: Using Dataset Labels (Default)

```python
import dtlpy as dl

# Get your dataset and model
dataset = dl.datasets.get(dataset_id='your-dataset-id')
model = dl.models.get(model_name='owlv2-huggingface-model')

# Model configuration (default uses dataset labels)
# No configuration changes needed!

# Run prediction on a dataset
model.predict_dataset(dataset=dataset)
```

### Example 2: Using Custom Labels

```python
import dtlpy as dl

model = dl.models.get(model_name='owlv2-huggingface-model')

# Update configuration to use custom labels
model.configuration['label_source'] = 'custom'
model.configuration['custom_labels'] = ['elephant', 'giraffe', 'zebra', 'lion']
model.update()

# Run prediction
items = dataset.items.list()
model.predict_items(items=items)
```

### Example 3: Adjusting Confidence Threshold

```python
import dtlpy as dl

model = dl.models.get(model_name='owlv2-huggingface-model')

# Lower threshold for more detections (may include false positives)
model.configuration['confidence_threshold'] = 0.2

# Higher threshold for fewer, more confident detections
model.configuration['confidence_threshold'] = 0.7

model.update()
```

### Example 4: Using a Larger Model

```python
import dtlpy as dl

model = dl.models.get(model_name='owlv2-huggingface-model')

# Use the large model for better accuracy
model.configuration['model_checkpoint'] = 'google/owlv2-large-patch14'
model.configuration['custom_labels'] = ['person', 'car', 'truck']
model.configuration['label_source'] = 'custom'
model.update()

# Run prediction
items = dataset.items.list()
model.predict_items(items=items)
```

## Tips and Best Practices

### 1. Text Query Formulation
- **Be specific**: "golden retriever" works better than just "dog"
- **Use common terms**: "car" works better than "automobile"
- **Try variations**: If "smartphone" doesn't work well, try "phone" or "mobile phone"
- **Keep it short**: Single words or short phrases work best

### 2. Confidence Threshold Tuning
- **Default (0.5)**: Good balance for most cases
- **Lower (0.2-0.3)**: More detections, useful for finding rare objects, but more false positives
- **Higher (0.6-0.8)**: Fewer false positives, cleaner results, but may miss objects

### 3. Performance Optimization
- **Batch processing**: Process multiple images at once for better throughput
- **Limit queries**: Too many text queries (>50) may slow down inference
- **Use base model**: Start with `google/owlv2-base-patch16` unless you need maximum accuracy


## Limitations

- **Zero-shot nature**: May not be as accurate as fine-tuned models for specific domains
- **Text dependency**: Detection quality depends on how well queries describe objects
- **Computational cost**: Larger models and more queries increase processing time
- **Language**: Best performance with English text queries

## Support

For issues, questions, or feature requests:
- [GitHub Repository](https://github.com/dataloop-ai-apps/huggingface-adapter)
- [Dataloop Documentation](https://docs.dataloop.ai/)
- [OWLv2 Paper](https://arxiv.org/abs/2306.09683)
