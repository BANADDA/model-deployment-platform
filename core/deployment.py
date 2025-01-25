# core/deployment.py

import uuid
from typing import Dict, Any
from kubernetes import client, config
import docker

class Deployment:
    def __init__(self, id: str, access_token: str):
        self.id = id
        self.access_token = access_token

class DeploymentManager:
    def __init__(self):
        self.docker_client = docker.from_env()
        config.load_kube_config()
        self.k8s_client = client.CoreV1Api()
        
    async def deploy(self, container_id: str, machine_config: Dict[str, Any], endpoint_type: str) -> Deployment:
        deployment_id = str(uuid.uuid4())
        access_token = str(uuid.uuid4())
        
        # Create Kubernetes deployment
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=f"model-{deployment_id}"),
            spec=client.V1DeploymentSpec(
                replicas=1,
                selector=client.V1LabelSelector(
                    match_labels={"app": f"model-{deployment_id}"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": f"model-{deployment_id}"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name=f"model-{deployment_id}",
                                image=container_id,
                                resources=client.V1ResourceRequirements(
                                    requests=machine_config.get("resources", {})
                                )
                            )
                        ]
                    )
                )
            )
        )
        
        self.k8s_client.create_namespaced_deployment(
            namespace="default",
            body=deployment
        )
        
        return Deployment(deployment_id, access_token)

    async def get_status(self, deployment_id: str) -> str:
        try:
            deployment = self.k8s_client.read_namespaced_deployment(
                name=f"model-{deployment_id}",
                namespace="default"
            )
            return deployment.status.phase
        except client.ApiException:
            return "not_found"

    async def delete(self, deployment_id: str):
        try:
            self.k8s_client.delete_namespaced_deployment(
                name=f"model-{deployment_id}",
                namespace="default"
            )
        except client.ApiException:
            pass