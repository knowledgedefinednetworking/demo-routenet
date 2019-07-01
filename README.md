# Challenging the generalization capabilities of Graph Neural Networks for network modeling
### Demo paper
#### J. Suárez-Varela, S. Carol-Bosch, K. Rusek, P. Almasan, M. Arias, P. Barlet-Ros, A. Cabellos-Aparicio.
 
## Abstract
Today, network operators still lack functional network models able to make accurate predictions of end-to-end Key Performance Indicators (e.g., delay or jitter) at limited cost. Recently, a novel Graph Neural Network (GNN) model called RouteNet was proposed as a cost-effective alternative to estimate the per-source/destination pair mean delay and jitter in networks. Thanks to its GNN architecture that operates over graph-structured data, RouteNet revealed an unprecedented ability to learn and model the complex relationships among topology, routing and input traffic in networks. As a result, it was able to make performance predictions with similar accuracy than resource-hungry packet-level simulators even in network scenarios unseen during training. In this demo, we will challenge the generalization capabilities of RouteNet with more complex scenarios, including larger topologies with a wider variety of routing configurations and traffic intensities than in the original work's evaluation.
 
<!-- Add BibTex citation to paper -->
 
## Description
In this demo, we extended the original implementation of RouteNet to support different link capacities. All the code we use in the demo is in the [Code directory](Code).
 
All the datasets used in this paper are available [here](https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v0), where we provide a detailed description on how to process the data.
 
Moreover, we provide a RouteNet model already trained (in the [models_trained directory](trained_models)) that can be directly loaded to make delay predictions on any sample from our datasets. In particular, this model was trained over 480,000 training samples: 240,000 from the 14-nodes NSF network topology and 240,000 from a 50-node network topology (see mode details in the description of our [datasets](https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v0)).
 
 
## Running RouteNet 
In order to make it easier the execution of RouteNet we provide some example functions in the **run_routenet.sh** script. This script contains different calls to the code of RouteNet (**routenet_with_link_cap.py**). To this end, we provide some predefined hyperparameters that can be easily modified in the script.
