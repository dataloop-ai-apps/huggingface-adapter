import os
import dtlpy as dl
from model_adapter import ModelAdapter


def package_creation(project):
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={'weights_filename': 'huggingface.pt',
                                                                 'epochs': 10,
                                                                 'batch': 4,
                                                                 'top_k': 5,
                                                                 'device': 'cuda:0'}
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')
    package = project.packages.push(package_name='hugging-face',
                                    ignore_sanity_check=True,
                                    src_path=os.getcwd(),
                                    # description='Global Dataloop ResNet implemented in pytorch',
                                    # scope='public',
                                    package_type='ml',
                                    # codebase=dl.GitCodebase(git_url='https://github.com/dataloop-ai/pytorch_adapters',
                                    #                         git_tag='mgmt3'),
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_S,
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=1,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)
    # s = package.services.list().items[0]
    # s.package_revision = package.version
    # s.versions['dtlpy'] = '1.63.2'
    # s.update(True)
    print("Package created!")
    return package


if __name__ == "__main__":
    env = 'prod'
    project_name = 'nlp-experiments'
    project_id = "e370fbba-7b8a-4f72-abf3-52b079f019e9"
    dl.setenv(env)
    project = dl.projects.get(project_id=project_id)
    package = package_creation(project)
    for req in package.requirements:
        print(f"{req.name} {req.operator} {req.version}")
