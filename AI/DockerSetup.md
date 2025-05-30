<h1>SIWN – AI Docker Container (ARM and x86) </h1>

Secure Infrastructure-less Wireless Networks (SIWNs) is a technology that allows a set of clients to exchange information securely through an ad-hoc wireless network. SIWNs provide security through client authentication, zero knowledge key generation for data encryption, and ensuring network trust through blockchain enabled network mapping. 

This document includes the structure and requirements for the development of a docker container that contains the necessary requirements for the deployment of the AI component of the SIWN 

<h3>References </h3>

* SIWN AI module docker containers: [ARM](https://hub.docker.com/repository/docker/joseasanchezviloria/siwn-ai/general) | [x64](https://hub.docker.com/layers/joseasanchezviloria/my-repo/tensorflow-CAAI-FAU-HPC/images/sha256-359a14f949900b1e40539eda574afb32e5f5c2c6f969807306d2ba7e74acc330?context=repo)

* [Docker containers](https://docs.docker.com/guides/walkthroughs/what-is-a-container/)

* [SIWN AI repo](https://github.com/C2A2-at-Florida-Atlantic-University/siwn-node/tree/main/AI)

* [Jetson Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/product-development/)

* [Tensorflow](https://www.tensorflow.org)

<h3>Background </h3>

* Docker: Docker is an open-source platform for developing, deploying, and running applications. It uses a concept called containerization to package up an application with all its dependencies into standardized units called containers. 

* Container: Containers have everything that your code needs in order to run, down to a base operating system. 

<h3>Use cases </h3>
The SIWN AI docker container can run the software that enables the AI-assisted signal processing operations of a SIWN node. This includes all software to deploy a complete AI solution. This includes AI-assisted operations such as physical layer fingerprinting and modulation classification. 

<h3>Requirements </h3>

* This should run in an embedded system 

* Operating systems: Ubuntu 20, MacOS, and x86
  
* It should be compatible with the use of GPUs when available
 
* Using common AI development software such as TensorFlow
  
* Able to communicate with other commponets through some interface (i.e API, sockets, ...)
 
<h3>Content </h3>
To perform the operations to facilitate certain signal processing operations, we will be developing our own custom AI-assisted signal processing tools through open-source software such as TensorFlow. TensorFlow is a free and open-source software library for machine learning and artificial intelligence that facilitates the development of CNN while supporting the use of GPUs. The main programming language that will be used for this development is Python 3.8. Further development could include using lower-level languages that support the TensorFlow library such as C++. As of May 25, 2024, these operations have been written in python and can be found in the following [Git Repository](https://github.com/C2A2-at-Florida-Atlantic-University/siwn-node/tree/main/AI). 

To enable the development, training, and testing of the CNNs that enable the AI-assisted operations we need the environment with all dependencies and resources for having a smooth development and deployment process. We have developed two environments. The ARM environment is for the deployment in embedded systems and the Linux environment for deployment in systems that enable accelerated development, training, and testing of CNNs. The Ubuntu version in the ARM environment is 20.04 and the version for the x86 is 22.04 

The docker container will be available in Jose Sanchez docker repos ([ARM](https://hub.docker.com/repository/docker/joseasanchezviloria/siwn-ai/general) | [x84](https://hub.docker.com/layers/joseasanchezviloria/my-repo/tensorflow-CAAI-FAU-HPC/images/sha256-359a14f949900b1e40539eda574afb32e5f5c2c6f969807306d2ba7e74acc330?context=repo))and should at some point be under some common repo for this project. The following are some operations for pulling, running, starting, and executing the docker container. 

<b>Pull Docker Container  </b>

* X86: docker image pull joseasanchezviloria/my-repo:tensorflow-CAAI-FAU-HPC
* ARM: docker image pull joseasanchezviloria/siwn-ai:v1 
 
<b>Run docker container: </b> 

* X86: docker run --name siwn-ai --runtime nvidia -it -e DISPLAY=$DISPLAY -v ~/siwn:/home/siwn -v /dev/net/tun:/dev/net/tun --network host --cap-add=NET_ADMIN joseasanchezviloria/my-repo:tensorflow-CAAI-FAU-HPC 
* ARM: docker run --name siwn-ai --runtime nvidia -it -e DISPLAY=$DISPLAY -v ~/siwn:/home/siwn -v /dev/net/tun:/dev/net/tun --network host --cap-add=NET_ADMIN joseasanchezviloria/siwn-ai:v1 
 

<b>Start Docker container:  </b>

docker start siwn-ai 

<b>Stop docker container:  </b>

docker stop siwn-ai 

<b>Execute (access) Docker container </b> 

docker exec –it siwn-ai /bin/bash 

<h3>Available development resources </h3>

The [CAAI lab](https://www.fau.edu/engineering/research/c2a2/) has access to resource with High Performance Computing (HPC) which are accessible through [FAU HPC cluster](https://helpdesk.fau.edu/TDClient/2061/Portal/Requests/ServiceCatalog?CategoryID=1480) and access can be requested by [Dr. George Sklivanitis](https://www.fau.edu/engineering/directory/faculty/sklivanitis/). To run Docker containers at the CAAI HPC, you must request to be added to the docker group.  

Use the instructions in the following document with instructions to [access CAAI HPC resources](https://github.com/C2A2-at-Florida-Atlantic-University/Setup-Guides/blob/main/HPC-Access.md).

<h3>Other approaches considered </h3>
It was also considered to create environments instead of containers, but containers are a more suitable approach since we needed to containerize OS specific dependencies other than just python or language specific dependencies. 
