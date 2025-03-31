import dtlpy as dl
from model_adapter import ModelAdapter


def run_train(env, project_name, dataset_name, model_name, delete_previous_model=False):
    dl.setenv(env)
    project = dl.projects.get(project_name=project_name)
    dataset = project.datasets.get(dataset_name=dataset_name)
    model = project.models.get(model_name=model_name)

    model.metadata['system'] = {}
    model.metadata['system']['subsets'] = {}

    train_filters = dl.Filters(field='metadata.system.tags.train', values=True)
    val_filters = dl.Filters(field='metadata.system.tags.validation', values=True)

    model.metadata['system']['subsets']['train'] = train_filters.prepare()
    model.metadata['system']['subsets']['validation'] = val_filters.prepare()
    
    model.output_type = 'text'

    try:
        if delete_previous_model:
            prev_model = project.models.get(model_name=model_name+'_SFT')
            print(f"Found previous model: {prev_model.name}, model ID: {prev_model.id}")
            input("Are you sure you want to delete the previous model? (y/n)")
            if input() == 'y':
                prev_model.delete()
    except Exception as e:
        pass

    new_model_name = model.name + '_SFT'
    # replace 90b with 11b
    new_model_name = new_model_name.replace('90b', '11b')
    # creates new dummy model entity
    new_model = model.clone(model_name=new_model_name, dataset=dataset)

    app = ModelAdapter()
    app.train_model(new_model)


if __name__ == '__main__':
    ENV = "rc"
    # PROJECT_NAME = "test yaya"
    # DATASET_NAME = "Data Management Demo Dataset prompt items-o4pVA"
    # DATASET_NAME = "tiny taco prompt items"

    PROJECT_NAME = "Model mgmt demo"
    # DATASET_NAME = "clip testing"   
    DATASET_NAME = "ceyda_fashion prompt items"
    
    MODEL_NAME = "llama-3-2-90b-vision-instruct"

    run_train(env=ENV, project_name=PROJECT_NAME, dataset_name=DATASET_NAME, model_name=MODEL_NAME, delete_previous_model=True)
