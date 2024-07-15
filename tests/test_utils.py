import dtlpy as dl
from typing import List, Union
import json
import time
import uuid
import os
import pathlib
import argparse
import yaml  # Required Installation
import logging

logger = logging.getLogger(name='dtlpy')

BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
COMMIT_ID = os.environ['COMMIT_ID']


class TestUtils:
    def __init__(self, project: dl.Project, commit_id: str, root_path: str, test_path: str):
        self.project = project
        self.commit_id = commit_id
        self.identifier = str(uuid.uuid4())[:8]
        self.tag = f"{self.commit_id}-{self.identifier}"

        # Paths
        self.root_path = root_path
        self.test_path = test_path
        self.datasets_path = os.path.join(self.root_path, 'tests', 'assets', 'e2e_tests', 'datasets')
        self._setup()

    def _setup(self):
        dataloop_cfg_filepath = os.path.join(self.root_path, ".dataloop.cfg")
        config_yaml_filepath = os.path.join(self.test_path, "config.yaml")
        template_json_filepath = os.path.join(self.test_path, "template.json")

        # Validations
        if not os.path.exists(dataloop_cfg_filepath):
            raise ValueError(f"'.dataloop.cfg' file wasn't found in '{self.root_path}'")
        if not os.path.exists(config_yaml_filepath):
            raise ValueError(f"'config.yaml' file wasn't found in '{self.test_path}'")
        if not os.path.exists(template_json_filepath):
            raise ValueError(f"'template.json' file wasn't found in '{self.test_path}'")

        # Load '.dataloop.cfg'
        with open(dataloop_cfg_filepath, 'r') as f:
            self.dataloop_cfg = json.loads(f.read())

        # Load 'config.yaml'
        with open(config_yaml_filepath, 'r') as f:
            self.config_yaml = yaml.load(f, Loader=yaml.FullLoader)

        # Load 'template.json'
        with open(template_json_filepath, 'r') as f:
            self.template_json = json.load(f)

    def create_dataset(self, dataset_name: str) -> dl.Dataset:
        """
        Create a dataset with the given name and folder path.
        If the folder contains an 'ontology' folder, it will be uploaded as the dataset's ontology.
        If the folder contains an 'items' folder, it will be uploaded as the dataset's items.
        If the folder contains a 'json' folder, it will be uploaded as the annotations to the given dataset's items.
        """
        new_dataset_name = f"{dataset_name}-{self.tag}"

        # Create dataset
        dataset: dl.Dataset = self.project.datasets.create(dataset_name=new_dataset_name)

        # Get paths
        ontology_json_folder_path = os.path.join(self.datasets_path, dataset_name, 'ontology')
        items_folder_path = os.path.join(self.datasets_path, dataset_name, 'items')
        annotation_jsons_folder_path = os.path.join(self.datasets_path, dataset_name, 'json')

        # Upload ontology if exists
        if os.path.exists(ontology_json_folder_path) is True:
            ontology_json_filepath = list(pathlib.Path(ontology_json_folder_path).rglob('*.json'))[0]
            with open(ontology_json_filepath, 'r') as f:
                ontology_json = json.load(f)
            ontology: dl.Ontology = dataset.ontologies.list()[0]
            ontology.copy_from(ontology_json=ontology_json)

        # Upload items without metadata/tags and without annotations
        if os.path.exists(items_folder_path) is True and os.path.exists(annotation_jsons_folder_path) is False:
            items_path = os.path.join(items_folder_path, '*')  # Prevents creating directory
            dataset.items.upload(local_path=items_path)

        # Upload items with metadata/tags and annotations
        if os.path.exists(items_folder_path) is True and os.path.exists(annotation_jsons_folder_path) is True:
            item_binaries = sorted(list(filter(lambda x: x.is_file(), pathlib.Path(items_folder_path).rglob('*'))))
            annotation_jsons = sorted(list(pathlib.Path(annotation_jsons_folder_path).rglob('*.json')))

            # Validations
            if len(item_binaries) != len(annotation_jsons):
                raise ValueError(
                    f"Number of items ({len(item_binaries)}) "
                    f"is not equal to number of annotations ({len(annotation_jsons)})"
                )

            # Upload each item with its related json
            for item_binary, annotation_json in zip(item_binaries, annotation_jsons):
                # Load annotation json
                with open(annotation_json, 'r') as f:
                    annotation_data = json.load(f)

                # Extract tags
                item_metadata = dict()
                tags_metadata = annotation_data.get("metadata", dict()).get("system", dict()).get('tags', None)
                if tags_metadata is not None:
                    item_metadata.update({"system": {"tags": tags_metadata}})

                # Extract metadata (outside of system)
                for key, value in annotation_data.get("metadata", dict()).items():
                    if key not in ["system"]:
                        item_metadata.update({key: value})

                # Upload item
                dataset.items.upload(
                    local_path=str(item_binary),
                    local_annotations_path=str(annotation_json),
                    item_metadata=item_metadata
                )

        return dataset

    def publish_dpk_and_install_app(self, dpk_name: str, install_app: bool = True) -> (dl.Dpk, dl.App):
        new_dpk_name = f"{dpk_name}-{self.tag}"

        # Find dpk json
        dpk_json = None
        dpk_json_filepath = None
        for manifest in self.dataloop_cfg.get("manifests", list()):
            dpk_json_filepath = manifest
            with open(dpk_json_filepath, 'r') as f:
                dpk_json = json.load(f)
            if dpk_json["name"] == dpk_name:
                break
            dpk_json = None
            dpk_json_filepath = None

        # Throw error if dpk not found
        if dpk_json is None:
            raise ValueError(f"Could not find dpk with name '{dpk_name}' in '.dataloop.cfg' file")

        # Update the dpk
        dpk = dl.Dpk.from_json(_json=dpk_json, client_api=dl.client_api, project=self.project)
        dpk.name = new_dpk_name
        dpk.display_name = dpk.name
        dpk.scope = "project"
        dpk.codebase = None

        # TODO: check if triggers need to be renamed

        # Set directory to dpk directory
        dpk_dir = os.path.join(self.root_path, os.path.dirname(dpk_json_filepath))
        os.chdir(dpk_dir)

        # Publish dpk and install app
        dpk = self.project.dpks.publish(dpk=dpk)
        app = None
        if install_app is True:
            app = self.project.apps.install(dpk=dpk)

        # Return to original directory
        os.chdir(self.root_path)
        return dpk, app

    @staticmethod
    def _build_filters(filters_resource: dl.FiltersResource,
                       dpk: dl.Dpk = None, app: dl.App = None, component_name: str = None) -> dl.Filters:
        # Build filters
        filters = dl.Filters(resource=filters_resource)
        if dpk is not None:
            filters.add(field="app.dpkId", values=dpk.id)
        if app is not None:
            filters.add(field="app.id", values=app.id)
        if component_name is not None:
            filters.add(field="app.componentName", values=component_name)
        return filters

    def get_datasets(self, dpk: dl.Dpk = None, app: dl.App = None, component_name: str = None) -> List[dl.Dataset]:
        # Build filters
        filters = self._build_filters(
            filters_resource=dl.FiltersResource.DATASET,
            dpk=dpk, app=app, component_name=component_name
        )

        # Get datasets
        datasets = self.project.datasets.list(filters=filters)
        if isinstance(datasets, dl.entities.PagedEntities):
            datasets = list(datasets.all())
        return datasets

    def get_models(self, dpk: dl.Dpk = None, app: dl.App = None, component_name: str = None) -> List[dl.Model]:
        # Build filters
        filters = self._build_filters(
            filters_resource=dl.FiltersResource.MODEL,
            dpk=dpk, app=app, component_name=component_name
        )

        # Get models
        models = self.project.models.list(filters=filters)
        if isinstance(models, dl.entities.PagedEntities):
            models = list(models.all())
        return models

    def get_services(self, dpk: dl.Dpk = None, app: dl.App = None, component_name: str = None) -> [dl.Service]:
        # Build filters
        filters = self._build_filters(
            filters_resource=dl.FiltersResource.SERVICE,
            dpk=dpk, app=app, component_name=component_name
        )

        # Get service
        services = self.project.services.list(filters=filters)
        if isinstance(services, dl.entities.PagedEntities):
            services = list(services.all())
        return services

    def _create_pipeline_from_json(self, pipeline_json: dict, install_pipeline: bool = True) -> dl.Pipeline:
        new_pipeline_name = f'{pipeline_json["name"]}-{self.tag}'[:35]

        # Update pipeline template
        pipeline_json["name"] = new_pipeline_name
        pipeline_json["projectId"] = self.project.id
        pipeline = self.project.pipelines.create(pipeline_json=pipeline_json)

        if install_pipeline:
            pipeline = pipeline.install()
        return pipeline

    def create_pipeline(self, pipeline_template_dpk: dl.Dpk, install_pipeline: bool = True) -> dl.Pipeline:
        # Get pipeline template from dpk
        pipeline_json = pipeline_template_dpk.components.pipeline_templates[0]
        return self._create_pipeline_from_json(pipeline_json=pipeline_json, install_pipeline=install_pipeline)

    def create_test_pipeline(self, install_pipeline: bool = True) -> dl.Pipeline:
        return self._create_pipeline_from_json(pipeline_json=self.template_json, install_pipeline=install_pipeline)

    @staticmethod
    def update_pipeline_variable(pipeline: dl.Pipeline, variables_dict: dict) -> dl.Pipeline:
        variable_keys = list(variables_dict.keys())
        variable: dl.Variable
        # TODO: Check if this exists in the SDK
        for variable in pipeline.variables:
            if variable.name in variable_keys:
                variable.value = variables_dict[variable.name]
        pipeline = pipeline.update()
        return pipeline

    @staticmethod
    def pipeline_execution_wait(pipeline_execution: dl.PipelineExecution) -> dl.PipelineExecution:
        pipeline = pipeline_execution.pipeline
        in_progress_statuses = ["pending", "in-progress"]
        while pipeline_execution.status in in_progress_statuses:
            time.sleep(5)
            pipeline_execution = pipeline.pipeline_executions.get(pipeline_execution_id=pipeline_execution.id)
        return pipeline_execution


class TestResources:
    def __init__(self):
        self.dpks = dict()
        self.apps = dict()
        self.datasets = dict()
        self.models = dict()
        self.services = dict()
        self.pipelines = dict()


class TestRunner:
    def __init__(self, test_args: argparse.Namespace):
        # Read args
        self.root_path = os.path.abspath(test_args.root_path)
        self.test_path = os.path.abspath(test_args.test_path)

        # Define variables
        self.dpks_creation_order = list()
        self.test_resources = TestResources()

        # Setup Test Utils
        self._init_test_utils()

    def _init_test_utils(self):
        # Login and get project
        dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        self.project = dl.projects.get(project_id=PROJECT_ID)
        self.commit_id = COMMIT_ID
        self.test_utils = TestUtils(
            project=self.project,
            commit_id=self.commit_id,
            root_path=self.root_path,
            test_path=self.test_path
        )

    # TODO: Add cleanup (waiting for decided test identifier)
    def _clean_up(self):
        pass

    @staticmethod
    def _get_key_value(entity_dict: Union[dict, str]) -> (str, dict):
        if isinstance(entity_dict, dict):
            key = list(entity_dict.keys())[0]
            value = entity_dict[key]
        elif isinstance(entity_dict, str):
            key = entity_dict
            value = dict()
        else:
            raise ValueError("Entity must be a dictionary or a string")
        return key, value

    def _prepare_dpks_and_apps(self):
        for dpk_entity in self.test_utils.config_yaml.get("dpks", list()):
            dpk_name, dpk_info = self._get_key_value(entity_dict=dpk_entity)
            install_app = dpk_info.get("install_app", False)

            # Validations
            if not isinstance(install_app, bool):
                raise ValueError("install_app must be a boolean value")

            # Publish dpk and install app
            dpk, app = self.test_utils.publish_dpk_and_install_app(dpk_name=dpk_name, install_app=install_app)
            self.test_resources.dpks.update({dpk_name: dpk})
            if app is not None:
                self.test_resources.apps.update({dpk_name: app})

            # Add dpk to creation order
            self.dpks_creation_order.append(dpk_name)

        # Reverse dpks order
        self.dpks_creation_order.reverse()

    def _prepare_local_datasets(self):
        for dataset_name in self.test_utils.config_yaml.get("local_datasets", list()):
            dataset = self.test_utils.create_dataset(dataset_name=dataset_name)
            self.test_resources.datasets.update({dataset_name: dataset})

    def _prepare_remote_datasets(self):
        for dataset_entity in self.test_utils.config_yaml.get("datasets", list()):
            dataset_name, dataset_info = self._get_key_value(entity_dict=dataset_entity)
            source_app = dataset_info.get("source_app", None)

            # Dataset created by dependency
            if source_app is None:
                datasets = self.test_utils.get_datasets(component_name=dataset_name)

                # Validations
                count = len(datasets)
                if count != 1:
                    raise ValueError(f"Expected 1 result for dataset name '{dataset_name}', but {count} were found")
                dataset = datasets[0]

                # Add dataset app to test resources
                app = self.project.apps.get(app_id=dataset.metadata['system']['app']['id'])
                self.test_resources.apps.update({app.name: app})
            # Dataset created by current installed apps
            else:
                # Get dataset app
                app = self.test_resources.apps.get(source_app, None)

                # Validations
                if app is None:
                    raise ValueError(f"App '{source_app}' was not found")

                # Find dataset in datasets
                datasets = self.test_utils.get_datasets(app=app, component_name=dataset_name)

                # Validations
                count = len(datasets)
                if count != 1:
                    raise ValueError(f"Expected 1 result for dataset name '{dataset_name}', but {count} were found")
                dataset = datasets[0]

            self.test_resources.datasets.update({dataset_name: dataset})

    def _prepare_models(self):
        for model_entity in self.test_utils.config_yaml.get("models", list()):
            model_name, model_info = self._get_key_value(entity_dict=model_entity)
            deploy_model = model_info.get("deploy_model", False)
            source_app = model_info.get("source_app", None)

            # Validations
            if not isinstance(deploy_model, bool):
                raise ValueError("deploy_model must be a boolean value")

            # Model created by dependency
            if source_app is None:
                models = self.test_utils.get_models(component_name=model_name)

                # Validations
                count = len(models)
                if count != 1:
                    raise ValueError(f"Expected 1 result for model name '{model_name}', but {count} were found")
                model = models[0]

                # Add model app to test resources
                app = self.project.apps.get(app_id=model.metadata['system']['app']['id'])
                self.test_resources.apps.update({app.name: app})
            # Model created by current installed apps
            else:
                # Get model app
                app = self.test_resources.apps.get(source_app, None)

                # Validations
                if app is None:
                    raise ValueError(f"App '{source_app}' was not found")

                # Find model in models
                models = self.test_utils.get_models(app=app, component_name=model_name)

                # Validations
                count = len(models)
                if count != 1:
                    raise ValueError(f"Expected 1 result for model name '{model_name}', but {count} were found")
                model = models[0]

            self.test_resources.models.update({model_name: model})

    def _prepare_services(self):
        for service_entity in self.test_utils.config_yaml.get("services", list()):
            service_name, service_info = self._get_key_value(entity_dict=service_entity)
            source_app = service_info.get("source_app", None)

            # Service created by dependency
            if source_app is None:
                services = self.test_utils.get_services(component_name=service_name)

                # Validations
                count = len(services)
                if count != 1:
                    raise ValueError(f"Expected 1 result for service name '{service_name}', but {count} were found")
                service = services[0]

                # Add service app to test resources
                app = self.project.apps.get(app_id=service.metadata['system']['app']['id'])
                self.test_resources.apps.update({app.name: app})
            # Service created by current installed apps
            else:
                # Get service app
                app = self.test_resources.apps.get(source_app, None)

                # Validations
                if app is None:
                    raise ValueError(f"App '{source_app}' was not found")

                # Find service in services
                services = self.test_utils.get_services(app=app, component_name=service_name)

                # Validations
                count = len(services)
                if count != 1:
                    raise ValueError(f"Expected 1 result for service name '{service_name}', but {count} were found")
                service = services[0]

            self.test_resources.services.update({service_name: service})

    def _prepare_pipelines(self):
        for pipeline_entity in self.test_utils.config_yaml.get("pipelines", list()):
            pipeline_template_dpk_name, pipeline_info = self._get_key_value(entity_dict=pipeline_entity)
            install_pipeline = pipeline_info.get("install_pipeline", False)

            # Validations
            if not isinstance(install_pipeline, bool):
                raise ValueError(f"install_pipeline must be a boolean value")

            # Create pipeline
            pipeline_template_dpk = self.test_resources.dpks.get(pipeline_template_dpk_name, None)

            # Validations
            if pipeline_template_dpk is None:
                raise ValueError(f"Dpk '{pipeline_template_dpk}' was not found")

            pipeline = self.test_utils.create_pipeline(
                pipeline_template_dpk=pipeline_template_dpk,
                install_pipeline=install_pipeline
            )
            self.test_resources.pipelines.update({pipeline_template_dpk_name: pipeline})

    def setup(self):
        self._prepare_dpks_and_apps()
        self._prepare_local_datasets()
        self._prepare_remote_datasets()
        self._prepare_models()
        self._prepare_services()
        self._prepare_pipelines()

    def _tear_down(self, test_pipeline: dl.Pipeline):
        test_pipeline.delete()

        pipeline: dl.Pipeline
        for pipeline in self.test_resources.pipelines.values():
            pipeline.delete()

        app: dl.App
        for app_name in self.dpks_creation_order:
            app = self.test_resources.apps.get(app_name, None)
            if app is not None:
                # Delete all app related models
                models = self.test_utils.get_models(app=app)
                for model in models:
                    model.delete()
                app.uninstall()

        dpk: dl.Dpk
        for dpk_name in self.dpks_creation_order:
            dpk = self.test_resources.dpks.get(dpk_name, None)
            if dpk is not None:
                dpk.delete()

        dataset: dl.Dataset
        for dataset in self.test_resources.datasets.values():
            dataset.delete(sure=True, really=True)

        dl.logout()

    def _search_entity_in_resources(self, entity_name: str, resource_type: str) -> any:
        available_resources = self.test_resources.__dict__
        if resource_type in list(available_resources.keys()):
            entity = available_resources[resource_type].get(entity_name, None)
        else:
            raise ValueError(f"Resource type '{resource_type}' is not supported")

        return entity

    def _parse_single_variable_value(self, variable_name: str, value_entity: dict) -> any:
        # Validations
        if value_entity is None:
            raise ValueError(f"Value for variable {variable_name} is not provided")

        variable_value_entity_name, variable_value_entity_info = self._get_key_value(entity_dict=value_entity)
        resource_type = variable_value_entity_info.get("resource_type", None)

        # Get entity
        entity = self._search_entity_in_resources(entity_name=variable_value_entity_name, resource_type=resource_type)

        # Validations
        if entity is None:
            raise ValueError(f"Entity '{variable_value_entity_name}' wasn't found in '{resource_type}'")

        # Get entity field (Default is 'id')
        entity_field = variable_value_entity_info.get("entity_field", "id")
        entity_value = getattr(entity, entity_field)
        return entity_value

    def _parse_variables(self, variables: dict) -> dict:
        variables_dict = dict()
        for variable_name, variable_info in variables.items():
            # Parse variable single value
            if isinstance(variable_info, dict):
                value_entity = variable_info
                variable_value = self._parse_single_variable_value(
                    variable_name=variable_name,
                    value_entity=value_entity
                )
                variable_dict = {variable_name: variable_value}
                variables_dict.update(variable_dict)
            # Parse variable list of values
            elif isinstance(variable_info, list):
                variable_value_list = list()
                for value_entity in variable_info:
                    variable_value = self._parse_single_variable_value(
                        variable_name=variable_name,
                        value_entity=value_entity
                    )
                    variable_value_list.append(variable_value)
                variable_dict = {variable_name: variable_value_list}
                variables_dict.update(variable_dict)
            # Not supported
            else:
                raise ValueError(f"Variable '{variable_name}' value type is not supported")

        return variables_dict

    def run(self):
        test_pipeline: dl.Pipeline = self.test_utils.create_test_pipeline(install_pipeline=False)

        # Update variables
        variables = self.test_utils.config_yaml.get("variables", dict())
        variables_dict = self._parse_variables(variables=variables)
        test_pipeline = self.test_utils.update_pipeline_variable(pipeline=test_pipeline, variables_dict=variables_dict)

        # Execute pipeline
        test_pipeline.install()
        pipeline_execution = test_pipeline.execute(execution_input=None)
        pipeline_execution = self.test_utils.pipeline_execution_wait(pipeline_execution=pipeline_execution)

        # Validate pipeline execution
        if pipeline_execution.status == dl.ExecutionStatus.SUCCESS.value:
            self._tear_down(test_pipeline=test_pipeline)
            logger.info(f"Test passed successfully!")
        else:
            raise ValueError(f"Pipeline failed with status '{pipeline_execution.status}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests Runner')
    parser.add_argument('--root_path', type=str, default="./", metavar='N',
                        help='Path of the test directory')
    parser.add_argument('--test_path', type=str, default=None, metavar='N',
                        help='Path of the test directory')
    args = parser.parse_args()

    # Set current working directory as root path
    if os.path.exists(args.root_path):
        os.chdir(args.root_path)
    else:
        raise ValueError(f"Path {args.root_path} does not exist")

    if args.test_path is None or not os.path.exists(args.test_path):
        raise ValueError(f"Path {args.test_path} does not exist or was not provided")

    # Run test
    test_runner = TestRunner(test_args=args)
    test_runner.setup()
    test_runner.run()
