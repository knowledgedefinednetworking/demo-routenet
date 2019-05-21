# Challenging the generalization capabilities of Graph Neural Networks for network modeling
### Demo paper
#### J. Suárez-Varela, S. Carol-Bosch, K. Rusek, P. Almasan, M. Arias, P. Barlet-Ros, A. Cabellos-Aparicio.

## Abstract
Today, network operators still lack functional network models able to make accurate predictions of end-to-end Key Performance Indicators (e.g., delay or jitter) at limited cost. Recently, a novel Graph Neural Network (GNN) model called RouteNet was proposed as a cost-effective alternative to estimate the per-source/destination pair mean delay and jitter in networks. Thanks to its GNN architecture that operates over graph-structured data, RouteNet revealed an unprecedented ability to learn and model the complex relationships among topology, routing and input traffic in networks. As a result, it was able to make performance predictions with similar accuracy than resource-hungry packet-level simulators even in network scenarios unseen during training. In this demo, we will challenge the generalization capabilities of RouteNet with more complex scenarios, including larger topologies with a wider variety of routing configurations and traffic intensities than in the original work's evaluation.

The source code, the delay model already trained and the training/evaluation datasets used in this paper are available [here](https://github.com/knowledgedefinednetworking/net2vec/tree/RouteNet-large-topologies).
