# Azure AI & IoT services
![alt text](https://github.com/mozamani/ai_iot/blob/master/files/logo.png)
------

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


4) [ONNX Runtime](https://github.com/microsoft/onnxruntime?WT.mc_id=iot-c9-niner) is a performance-focused complete scoring engine for [ONNX](https://onnx.ai/) models, with an open extensible architecture to continually address the latest developments in AI and Deep Learning. ONNX Runtime stays up to date with the ONNX standard with complete implementation of all ONNX operators, and supports all ONNX releases (1.2+) with both future and backwards compatibility.
![alt text](https://github.com/microsoft/onnxruntime/raw/master/docs/images/ONNX_Runtime_logo_dark.png)

5) [ONNX model Zoo](https://github.com/onnx/models)  is a collection of pre-trained, state-of-the-art models in the (ONNX) format contributed by community members like you. Accompanying each model are Jupyter notebooks for model training and running inference with the trained model. The notebooks are written in Python and include links to the training dataset as well as references to the original paper that describes the model architecture.
![alt text](https://github.com/onnx/models/blob/master/resource/images/ONNX%20Model%20Zoo%20Graphics.png)

6) Select ONNX examples: <br>
- [Object Detection with ONNX Runtime (YOLOv3)](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/yoloV3_object_dection_onnxruntime_inference.ipynb) <br>
- [YOLO Real-time Object Detection using ONNX on AzureML](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/yoloV3_object_dection_onnxruntime_inference.ipynb) <br>
- [Gallery of ONNX Runtime examples](https://microsoft.github.io/onnxruntime/auto_examples/index.html)

7) [This tutorial](https://github.com/Azure-Samples/onnxruntime-iot-edge) shows how to integrate Azure services with machine learning on the NVIDIA Jetson Nano (an ARM64 device) using Python. By the end of this sample, you will have a low-cost DIY solution for object detection within a space and a unique understanding of integrating ARM64 platform with Azure IoT services and machine learning.
![alt text](https://github.com/Azure-Samples/onnxruntime-iot-edge/raw/master/images_for_readme/arch.jpg)

8) [Real-time computer vision with Databricks Runtime for Machine Learning](https://databricks.com/blog/2018/09/13/identify-suspicious-behavior-in-video-with-databricks-runtime-for-machine-learning.html)
![alt text](https://databricks.com/wp-content/uploads/2018/09/db-video-pipeline.png)

9) Microsoft IoT channel demo video for ML deployment on edge: [Train with Azure ML and deploy everywhere with ONNX Runtime](https://www.youtube.com/watch?time_continue=409&v=JpfZxRsLgWg)<br>

10) Microsoft and NVIDIA extend video analytics to the intelligent edge using NVIDIA [DeepStream](https://developer.nvidia.com/deepstream-sdk) [link]((https://azure.microsoft.com/en-us/blog/microsoft-and-nvidia-extend-video-analytics-to-the-intelligent-edge/)
