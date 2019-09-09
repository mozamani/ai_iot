# Azure AI & IoT services

![alt text](https://github.com/mozamani/ai_iot/blob/master/files/logo.png) <!-- .element height="10%" width="10%" -->

## Deploying AI models on IoT Edge
![alt text](https://github.com/mozamani/ai_iot/blob/master/files/architecture.png)

### Computer Vision

1) This [tutorial](https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-deploy-custom-vision) shows how to train and deploy a comoputer vision model on IoT edge devices using the [Custom Vision](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/home) service. <br>
![alt text](https://docs.microsoft.com/en-us/azure/iot-edge/media/tutorial-deploy-custom-vision/custom-vision-architecture.png) 

2) Microsoft [Computer Vision](https://github.com/microsoft/ComputerVision) provides examples and best practice guidelines for building computer vision systems. All examples are given as Jupyter notebooks, and use PyTorch as the deep learning library.
![alt text](https://github.com/microsoft/ComputerVision/blob/master/media/intro_od_vis.jpg) <br>

3) Reference architectures for distributed training of deep learning models (on GPU) - this is an architecture for data-parallel distributed training with synchronous updates using [**Horovod**](https://github.com/horovod/horovod).<br> 
- [reference architecture](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/training-deep-learning)  
- [github page](https://github.com/microsoft/DistributedDeepLearning/)<br>
- Horovod [main paper](https://arxiv.org/pdf/1802.05799.pdf)<br>

![alt text](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/_images/distributed_dl_flow.png)

![alt text](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/_images/distributed_dl_architecture.png)

5) [This repository](https://github.com/microsoft/AKSDeploymentTutorialAML) provides a number of tutorials in Jupyter notebooks that have step-by-step instructions on how to deploy a pretrained deep learning model on a GPU enabled Kubernetes cluster throught Azure Machine Learning (AzureML)
![alt text](https://camo.githubusercontent.com/51f005d8fe7e49980f997b0350473c3fe3fe1a3b/68747470733a2f2f686170707970617468737075626c69632e626c6f622e636f72652e77696e646f77732e6e65742f616b736465706c6f796d656e747475746f7269616c616d6c2f617a757265696f746564676572756e74696d652e706e67)

4) [ONNX Runtime](https://github.com/microsoft/onnxruntime?WT.mc_id=iot-c9-niner) is a performance-focused complete scoring engine for [ONNX](https://onnx.ai/) models, with an open extensible architecture to continually address the latest developments in AI and Deep Learning. ONNX Runtime stays up to date with the ONNX standard with complete implementation of all ONNX operators, and supports all ONNX releases (1.2+) with both future and backwards compatibility.
![alt text](https://github.com/microsoft/onnxruntime/raw/master/docs/images/ONNX_Runtime_logo_dark.png)

5) [ONNX model Zoo](https://github.com/onnx/models)  is a collection of pre-trained, state-of-the-art models in the (ONNX) format contributed by community members like you. Accompanying each model are Jupyter notebooks for model training and running inference with the trained model. The notebooks are written in Python and include links to the training dataset as well as references to the original paper that describes the model architecture.
![alt text](https://github.com/onnx/models/blob/master/resource/images/ONNX%20Model%20Zoo%20Graphics.png)

6) Select ONNX examples: <br>
- [Tutorials](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx)  showing how to create and deploy Open Neural Network eXchange (ONNX) models in Azure Machine Learning environments using ONNX Runtime for inference <br>
- [Gallery of ONNX Runtime examples](https://microsoft.github.io/onnxruntime/auto_examples/index.html)

7) [This tutorial](https://github.com/Azure-Samples/onnxruntime-iot-edge) shows how to integrate Azure services with machine learning on the NVIDIA Jetson Nano (an ARM64 device) using Python. By the end of this sample, you will have a low-cost DIY solution for object detection within a space and a unique understanding of integrating ARM64 platform with Azure IoT services and machine learning.
![alt text](https://github.com/Azure-Samples/onnxruntime-iot-edge/raw/master/images_for_readme/arch.jpg)

8) Microsoft and [NVIDIA](https://developer.nvidia.com/deepstream-sdk)  extend video analytics to the intelligent edge using [DeepStream SDK](https://azure.microsoft.com/en-us/blog/microsoft-and-nvidia-extend-video-analytics-to-the-intelligent-edge/)
![alt text](https://azurecomcdn.azureedge.net/mediahandler/acomblog/media/Default/blog/e86d2867-40b5-4726-9334-82fb715526f5.jpg)

9) This [repo](https://github.com/Microsoft/vision-ai-developer-kit) contains the components needed to use the [Vision AI Developer Kit](https://azure.github.io/Vision-AI-DevKit-Pages/) to develop Neural Network models which can be deployed to the Vision AI DevKit hardware.
![alt text](https://azure.github.io/Vision-AI-DevKit-Pages/assets/images/Peabody_spec_image.png)


9) Microsoft IoT channel demo video for ML deployment on edge: [Train with Azure ML and deploy everywhere with ONNX Runtime](https://www.youtube.com/watch?time_continue=409&v=JpfZxRsLgWg)<br>

10) [Real-time computer vision with Databricks Runtime for Machine Learning](https://databricks.com/blog/2018/09/13/identify-suspicious-behavior-in-video-with-databricks-runtime-for-machine-learning.html)
![alt text](https://databricks.com/wp-content/uploads/2018/09/db-video-pipeline.png)



### MLOps with Azure ML
Azure ML contains a number of asset management and orchestration services to help you manage the lifecycle of your model training & deployment workflows.

With [Azure ML + Azure DevOps](https://github.com/Microsoft/MLOps) you can effectively and cohesively manage your datasets, experiments, models, and ML-infused applications.  
![alt text](https://github.com/microsoft/MLOps/raw/master/media/ml-lifecycle.png)


### Predictive Maintenance with AI
1) Deploy Azure Machine Learning predictive maintenance model as an IoT Edge module [demo link](https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-deploy-machine-learning) <br>

2) Batch scoring of Spark machine learning models [demo link](https://github.com/Azure/BatchSparkScoringPredictiveMaintenance) <br>

3) [Predictive Maintenance using PySpark](https://github.com/Azure/PySpark-Predictive-Maintenance) <br>

4) [Predictive Maintenance with AI](https://github.com/Azure/AI-PredictiveMaintenance) <br>

5) [Deep learning for predictive maintenance](https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb) <br>

6)



