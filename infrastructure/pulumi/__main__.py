# infrastructure/pulumi/__main__.py
import pulumi
import pulumi_aws as aws
import pulumi_kubernetes as k8s
from pulumi_kubernetes.helm.v3 import Chart, ChartOpts

# Configuration
config = pulumi.Config()
environment = config.require("environment")

# Create EKS cluster for production
if environment == "production":
    cluster = aws.eks.Cluster(
        "voice-doc-intelligence",
        instance_type="t3.medium",
        desired_capacity=3,
        min_size=1,
        max_size=5,
        node_associate_public_ip_address=False,
    )
    
    # Install NVIDIA GPU Operator for AI workloads
    gpu_operator = Chart(
        "gpu-operator",
        ChartOpts(
            chart="gpu-operator",
            version="v23.9.1",
            namespace="gpu-operator-resources",
            fetch_opts={"repo": "https://nvidia.github.io/gpu-operator"},
        ),
        opts=pulumi.ResourceOptions(provider=cluster.provider)
    )

# TigerGraph deployment
tigergraph_chart = Chart(
    "tigergraph",
    ChartOpts(
        chart="tigergraph",
        version="3.9.3",
        namespace="data",
        values={
            "global": {
                "storageClass": "gp3",
                "size": "100Gi"
            },
            "resources": {
                "requests": {
                    "memory": "8Gi",
                    "cpu": "2"
                }
            }
        }
    )
)

# Export cluster endpoint
pulumi.export("kubeconfig", cluster.kubeconfig)