# Azure AI & IoT services
![alt text](https://github.com/mozamani/ai_iot/blob/master/files/logo.png)
------

## Deploying AI models on IoT Edge
![alt text](https://github.com/mozamani/ai_iot/blob/master/files/architecture.png)

### Computer Vision

1) This [tutorial](https://docs.microsoft.com/en-us/azure/iot-edge/tutorial-deploy-custom-vision) shows how to train and deploy a comoputer vision model on IoT edge devices using the [Custom Vision](https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/home) service. <br>
![alt text](https://docs.microsoft.com/en-us/azure/iot-edge/media/tutorial-deploy-custom-vision/custom-vision-architecture.png) 

2) Microsoft[Computer Vision](https://github.com/microsoft/ComputerVision) provides examples and best practice guidelines for building computer vision systems. All examples are given as Jupyter notebooks, and use PyTorch as the deep learning library.
![alt text](https://github.com/microsoft/ComputerVision/blob/master/media/intro_od_vis.jpg) <br>

3) Reference architectures for distributed training of deep learning models (on GPU) - this is an architecture for data-parallel distributed training with synchronous updates using [**Horovod**](https://github.com/horovod/horovod).<br> 
-- [reference architecture](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/training-deep-learning)  
-- [github page](https://github.com/microsoft/DistributedDeepLearning/)
-- Horovod main [paper](https://arxiv.org/pdf/1802.05799.pdf)<br>

![alt text](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/_images/distributed_dl_flow.png)

![alt text](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/_images/distributed_dl_architecture.png)



3) [ONNX Runtime](https://github.com/microsoft/onnxruntime?WT.mc_id=iot-c9-niner) is a performance-focused complete scoring engine for [ONNX](https://onnx.ai/) models, with an open extensible architecture to continually address the latest developments in AI and Deep Learning. ONNX Runtime stays up to date with the ONNX standard with complete implementation of all ONNX operators, and supports all ONNX releases (1.2+) with both future and backwards compatibility.

### ONNX

3) Demo video: [Train with Azure ML and deploy everywhere with ONNX Runtime](https://www.youtube.com/watch?time_continue=409&v=JpfZxRsLgWg)
