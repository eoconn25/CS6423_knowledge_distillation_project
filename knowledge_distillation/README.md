# NAS + KD Algorithmic Implimentation

The key algorithmic experiment of this project will be centered on a NAS + KD process that aims to create an effective Student model to which we will distill our Teacher's knowledge.

This algorithm will be applied to several medical imaging models, including a ResNet10 backbone, ResNet18 backbone, ResNet50 backbone, and a Vision Transformer.  Below we will outline the specific steps that will be followed in the NAS + KD algorithm.

## NAS Idea

We are interested in model compression, in which the original Teacher is distilled to a smaller, more hardware-efficient Student.  As such, we require a hardware-aware approach to the NAS.  
* This steers us away from a cell-based search, which looks to optimize a small cell/block that is repeatedly stcked for the final architecture.
* We will thus attempt a **macro search** that trains an entire network from start to finish.  This leads to a more **hardware-aware/efficient** final result.

Search spaces for NAS, though, can be quite large.  Training all of these combinations of parameters would be extremely impractical.  As such, we will perform a **one-shot approach**, which trains a single "supernet" that encompasses the entire search space.

The idea is that, once our supernet is trained, we can then "turn off" different parts of the supernet, sampling some subnetwork that achieves the best performance.  **This will be our Student architecture.**

## NAS Practically
**1) Defining a Search Space**: The first step in our NAS. This search space will differ for each model to which we apply our NAS+KD algorithm.  These are the "levers" we will pull as we search for the optimal architecture for our student.

**2) Constructing a Supernet**: the supernet will encompass the entire search space, and will be the target of our knowledge distillation.  That is, we will distill the knowledge from our original ResNet/ViT Teacher to the supernet.

**3) Training the Supernet with GreedyNAS**: GreedyNAS is our Search Strategy, which will help us find the best architecture within our supernet.  
* The standard search strategy for a NAS process such as this would simply uniformly sample paths through our supernet to train.  That is, it would treat all paths equally, randomly sample some, update their weights, and then iterate.
* GreedyNAS, which we will use, improves on this by ignoring "weak paths" that are unlikely to be chosen in the final architecture.  It samples M paths in the supernet, evaluates their performance, and then only selects the k top-performing paths for weight updates.  This will save computation time and prevent weight updates on "weak paths" from distracting the "strong paths".

**4) Final Student Selection**: The final component of our NAS will be an **Evolutionary search**.  Supernet weights will be frozen (i.e., no more training will take place).  We will generate random valid paths through the netwrork, calculate a fitness score combining accuracy and latency metrics, and perform crossover on high-performing parents.  This will, after a number of iterations, yield our final Student architecture, fully distilled and ready for deployment.
* This will leverage a Hardware Lookup Table.  We will record latency and VRAM performance on our different individual components of the network, and evaluate the expected performance of our architecture to determine if it is "valid" for a student (within our hardware restrictions).

This defines the overall process of our NAS + KD algorithm.  We will now detail how the Supernet will be trained, taht is, how we will distill our knowledge from the pretrained Teacher (be it a ResNet or ViT) to the supernet.

## Supernet KD



