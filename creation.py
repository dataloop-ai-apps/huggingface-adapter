import os
import dtlpy as dl
from model_adapter import ModelAdapter


def package_creation(project: dl.Project, entry_point_path: str = 'model_adapter.py') -> dl.Package:
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={'weights_filename': 'huggingface.pt',
                                                                 'epochs': 10,
                                                                 'batch': 4,
                                                                 'top_k': 5,
                                                                 'device': 'cpu'}
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point=entry_point_path)
    package = project.packages.push(package_name='hugging-face',
                                    ignore_sanity_check=True,
                                    src_path=os.getcwd(),
                                    is_global=True,
                                    package_type='ml',
                                    modules=[modules],
                                    codebase=dl.GitCodebase(git_url='https://github.com/dataloop-ai-apps/huggingface-adapter.git',
                                                            git_tag='v0.1.12'),
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_M,
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)
    print("Package created!")
    return package


if __name__ == "__main__":
    env = 'prod'
    project_id = '<SET-PROJECT-ID>'
    dl.setenv(env)
    package = package_creation(dl.projects.get(project_id=project_id))
