**By: Ethan O’Connor**

**Contents**

Along with the other algorithmic experiments in this project, this component will be centered on a NAS + KD process that aims to create an effective Student model to which we will distill our Teacher's knowledge.

This algorithm, along with the others, will be applied to several medical imaging models, including a ResNet10 backbone, ResNet18 backbone, and ResNet50 backbone.  Below we will outline the specific steps that will be followed in the NAS + KD algorithm.

---

## I. General Methodology

### 1.1 Supernet Concept

We are interested in model compression, in which the original Teacher is distilled to a smaller, more hardware-efficient Student.  As such, we require a hardware-aware approach to the NAS.

- This steers us away from an exclusively cell-based search, which looks to optimize a small cell/block that is repeatedly stacked for the final architecture.
- We will thus attempt a **macro search** that trains an entire network from start to finish. This leads to a more **hardware-aware/efficient** final result.

Search spaces for NAS, though, can be quite large.  Training all of these combinations of parameters would be extremely impractical.  As such, we will perform a **one-shot approach**, which trains a single "supernet" that encompasses the entire search space.

The idea is that, once we distill our knowledge from the original teacher into the supernet, we can then "turn off" different parts of the supernet, sampling some subnetwork that achieves the best performance.  **This will be our Final Student architecture.**

During supernet training, different subpaths from the network are sampled for the forward pass and weight updates.  This is done to build meaningful, specialized subconfigurations.  Building such paths eases the stress of the final component of the methodology, in which all parts of the supernet are discarded aside from the optimal subconfiguration, which then becomes the final student architecture.

The standard search strategy for a supernet NAS process such as this would simply uniformly sample paths through our supernet to train. That is, it would treat all paths equally, randomly sample some, update their weights, and then iterate.

**GreedyNAS**, which we will use as part of our search strategy during supernet training, improves on this by ignoring "weak paths" that are unlikely to be chosen in the final architecture.  The algorithm samples M paths from the supernet, evaluates their performance, and then only selects the $k$ top-performing paths for weight updates. This will save computation time and prevent weight updates on "weak paths" from distracting the "strong paths" that are more likely to become the best student subconfiguration.

### 1.2 NAS+KD Procedure

For each model independently (ResNet50, ResNet18, ResNet10t), we will apply the following algorithm:

**1) Define a Search Space**: The first step in our NAS. This search space will differ for each model to which we apply our NAS+KD algorithm.  These are the "levers" we will pull as we search for the optimal architecture for our student.

**2) Construct a Supernet**: the supernet will encompass the entire search space, and will be the target of our knowledge distillation.  That is, we will distill the knowledge from our original ResNet50/18/10t Teacher to the supernet.  A separate supernet will be trained for each teacher model, since we are looking to perform three “trials” of our overall methodology.

**3) KD to train the Supernet with GreedyNAS**: The supernet will be trained using **response-based KD**, based on KL divergence loss with an alpha parameter that controls the soft loss vs. hard loss tradeoff, as well as a temperature parameter that scales the soft loss distribution.  This will incorporate the GreedyNAS approach as previously described.

**4) Select the Final Student**: The final component of our NAS will be an **Evolutionary search**.  Supernet weights will be frozen (i.e., no more training will take place).  We will generate random configurations of the network, calculate a fitness score promoting accuracy and penalizing size.  Crossover and mutation will be applied to high-performing parents.  This will, after a number of iterations, yield our final Student architecture, fully distilled and ready for deployment.

**5) Fine Tune:** We will extract the optimal configuration determined in our evolutionary search and perform some final fine-tuning using the original training/validation data.  This looks to simply provide an additional boost in the F1 score of our final student model.

This defines the overall process of our NAS + KD algorithm, which will yield a discrete, optimal subconfiguration of the supernet that will in turn be used as the final student architecture.  We will now discuss supernet training before further defining the evolutionary search procedure.

---

## II. Training the Supernet

### 2.1 Defining a Search Space

This is arguably the most important part of the entire procedure, as it defines different configurations that will be sampled from the supernet during training (by the GreedyNAS procedure) as well as during the evolutionary search.  Our final student, then, will be some configuration of the search space.

Our search space takes advantage of some of the existing patterns in ResNet architectures.  For example, the feature extraction elements of the ResNet50 can be understood as four distinct “stages”, each consisting of a sequence of convolutional and normalization layers.

![image.png](attachment:045d182d-d404-4f57-a281-2bdc86e669eb:image.png)

*Image source: https://www.ultralytics.com/blog/what-is-resnet-50-and-what-is-its-relevance-in-computer-vision*

Following this framework, our supernet architecture can likewise be composed of distinct stages, the following levers being altered in each stage as we sample subconfigurations from the network:

- Depth: the number of convolutional and pooling blocks in the stage.
- Width: the number of filters in our convolutional layers within the stage.
- Expansion Ratio: as evidenced by the diagram above, each stage in the ResNet50 has a bottleneck compressing the output of the previous stage into a smaller number of filters.  This “expansion ratio” of the input size to output size of each stage can likewise be a part of the search space.
    - This only applies to the ResNet50, though, as the simpler 18 and 10t models do not have this feature

We can also vary the input resolution of our original images.  The original (224x224) shape of our images can be interpolated to lower resolutions as needed.

While we need only select the input resolution once when defining a specific subconfiguration, we sample a depth, width, and expansion ration parameter for each stage.  For example, we would select 4 depths for the ResNet50’s supernet, one for each distinct stage, allowing us to vary the number of convolutional layers in each.

### 2.2 Weight Slicing

Our overall strategy of first training the supernet before “turning off” different regions to obtain a smaller student model is made possible via weight slicing.  With this, we are able to define the supernet at its maximum configuration (for example, its max width, depth, and expansion ratio) and then “slice” out the inner weights for smaller configurations. 

In other words, when we want to create a smaller configuration with the width of 0.75, for example, we don’t create an entirely new layer, but instead index the existing filters.  This extracts the top 75% of convolutional filters from the supernet, and uses only those to make a prediction.

In our project, this was implemented using custom convolutional and batch normalization layers, built on top of PyTorch’s 2D Convolutional module.  These layers are initialized using the maximum in channels and out channels.  The forward pass, however, is indexed to match the size of the input.  This is the mechanism used to adjust the number of filters present in the supernet’s convolutional layers on the fly, using less filters for smaller subconfigurations and all available filters for larger/maximal configurations.

![image.png](attachment:c71b7c3b-ba52-4110-82ab-8a60384f553b:image.png)

These custom Convolutional layers, along with custom Batch Normalization layers, can be stacked to form custom stages of our supernet architecture.

Although we do not search over the convolutional kernel size, the weight slicing can also be illustrated in the context of slicing a kernel.  We define the supernet to work with the max kernel size (5x5, for example).  If we wish to sample a path with a 3x3 kernel, we simply take the inner 3x3 segment of the original 5x5 kernel.

![image.png](attachment:d3c3930d-3ac0-497d-921a-d8977b9cc4cc:image.png)

*Image source: Li, et. al, DS-Net++: Dynamic Weight Slicing for Efficient Inference in CNNs and Transformers, https://arxiv.org/abs/2109.10060*

### 2.3 Supernet Architectures

With our search being performed over the aforementioned categories, we can design a specific search space for each teacher model.  The supernet for each model, then, would be defined as a ResNet-like architecture with the maximum value from each category.  Once this maximal network is defined, containing all of the possible combinations of the search space, we will perform weight slicing to extract subnetworks during GreedyNAS training and evolutionary search.

We define the search spaces for each model as follows.  The rationale for the differences in each is briefly explained.  However, due to time constraints on the project, these were decided on at a fairly early stage and with limited knowledge.  If the project were to be redone, a more strategic definition of these search spaces would be prudent, given their importance in directly determining the final result.

**ResNet50 Supernet**

- input resolution = [128, 160, 190, 224] select one
- depth = [2, 3, 4]
- width = [0.65, 0.8, 1.0, 1.2]
- expansion ratio = [3,4,6]

The default ResNet50 architecture would possess an input resolution of 224.  At each stage, the default architecture has a depth of 3, a width of 1.0, and an expansion ratio of 4.  The supernet architecture is defined at the maximum configuration of the search space, with an input resolution of 224 and across all stages a depth of 4, a width of 1.2, and an expansion ratio of 6.

When sampling subconfigurations, we select a different depth and expansion ratio for each of the four stages present in the ResNet50.  This allows us to randomly increase or decrease the expressive capabilities of each stage in our supernet - that is, final student architecture need not be homogeneous in its stage depth as the original ResNet50 is.  The depth of the first stage of the student architecture could be 4, for example, with other stages only having a depth of 2 or 3.

This elasticity allows our NAS to determine the expressive requirements for each stage, and adjust the student architecture as necessary to fit those requirements.  

**ResNet18 Supernet**

- input resolution = [128, 160, 190, 224]
- depth = [1, 2, 3]
- width = [0.5, 0.75, 1.0, 1.2]

Key differences between this and the ResNet50 supernet search space include the lack of the expansion ratio parameter.  ResNet18 architectures do not possess this bottleneck feature, so it is omitted here.  The width options are also slightly altered, to give the NAS the option to scale model width even smaller than was possible in the ResNet50 supernet search space.  Finally, a shallower depth of 1 is included instead of a depth of 4, to resemble the simpler architecture of the teacher ResNet18.

**ResNet10t Supernet**

- input resolution = [112, 128, 160, 224]
- depth = [1, 2]
- width = [0.5, 0.75, 1.0, 1.2]

This search space is quite similar to that of the ResNet18, with the only difference being that a lower input resolution (112) is made available.  This is to give the NAS an option to substantially decrease the input resolution of the original images, potentially making our student model competitive for extremely lightweight applications.  In other words, it is because I was curious if the NAS would prefer an even more microscopic image resolution than what is available in the other search spaces.

The depth option of 3 is likewise removed, as a depth of 2 would already double number of convolutions per stage of the ResNet18.  This seemed sufficient as a more-expressive option for the NAS without needing to include a depth = 3 option.

**Example**

To help illustrate how a subconfiguration is selected for each model type, this is a function similar to one in the actual code base that could be used to sample a subconfiguration from the ResNet50 supernet:

![image.png](attachment:855be8c0-133b-4cc0-9fbb-38f376c8b92b:2b1f769c-cfb6-4656-9bae-651163c8d0bf.png)

### 2.4 Supernet Loss Function

Our loss function for supernet training is a combination of the student’s hard loss (compared to the ground-truth labels from our dataset) and the soft loss, which compares the student’s predictions to the teacher’s.  This tradeoff is scaled by a parameter, $\alpha$.  We define our total Distillation Loss as:

$$
\mathcal{L}_{total}=\alpha(\mathcal{L}_{soft}) + (1-\alpha)(\mathcal{L}_{hard})
$$

Further focusing on these components, we specify the hard loss as standard Cross-Entropy, and use PyTorch’s existing function to implement it.  This existing function also allows us to leverage Label Smoothing, which forces the model to distribute its predictions a bit.  This is aimed to help minimize overconfidence and alleviate overfitting.  The standard Cross-Entropy loss is specified as:

$$
\mathcal{L}_{hard}=-\sum^N_{i=1}y_ilog(\hat{y}_{s,i})
$$

The soft loss is a bit more complex, as it relaxes the teacher’s prediction to soft targets (logits) and compares this to the student’s logits using Kullback-Leibler Divergence.  In our course materials on Canvas, the response-based soft targets are defined as probabilities estimated with a temperature factor $T$:

$$
p(z_i,T)=\frac{exp(z_i/T)}{\sum exp(z_j/T)}
$$

These soft targets are obtained for both the teacher and the student (they must be transformed to log-softmax for the built-in Pytorch KL function).  These are then fed to PyTorch’s KL Divergence function, quantifying the difference between our student and teacher’s prediction distributions.  We can thus express our soft loss as:

$$
\mathcal{L}_{soft}=T^2\cdot D_{KL}(Softmax(\frac{z_T}{T}),Softmax(\frac{z_s}{T}))
$$

This combination of hard and soft loss, scaled by $\alpha$, is implemented as follows in Python:

![image.png](attachment:67519f19-3a05-4c18-b384-e4b044ed19ca:image.png)

### 2.5 Training Considerations

The supernet is, by definition, overparameterized.  It includes the entire search space, which results in a massive, highly expressive architecture that is larger than the original teacher itself.  This would seem to make overfitting a likely issue.  However, it is our path sampling mechanism that prevents this from happening.

- For each epoch, we do not train the entire supernet, but a selection of subconfigurations.  This means that our overparameterized maximal path is very rarely sampled and updated.
- These subconfigurations are further filtered by the GreedyNAS algorithm: a random selection of subconfigurations are sampled, but GreedyNAS evaluates the performance of each and only updates the weights of the top $k$ paths.

For these reasons, the subpath sampling scheme helps us avoid overfitting our large, overparameterized model.

Another danger in supernet training is path inversion, where smaller subconfigurations (MIN paths) perform better than the larger MAX path.  Since the supernet is supposed to act as a proxy for the performance of many individual models, path inversion indicates that this proxy is breaking down.  Our supernet is no longer comparable to the training of independent, separate models, where the MAX model would always be expected to be more expressive than the MIN model.

- To help monitor this, the MAX and MIN paths are examined at each validation epoch.  If the MIN accuracy and F1 are significantly better than the MAX accuracy and F1, path inversion is present.
- If path inversion is present, our final search results may be biased toward sub-optimal small networks.

---

## III. Performing the Evolutionary Search

After we have trained the supernet using our previously described Distillation Loss, the knowledge from the student has been distilled into our supernet.  Furthermore, the GreedyNAS algorithm has helped us to develop high performing subpaths within our supernet.  We now need to select a smaller architecture out of the overparameterized supernet to serve as the final student architecture.  This component of the NAS+KD process is done via evolutionary search.

### 3.1 Search Space

Our search space for the evolutionary algorithm is the same as those defined above in Section 2.3.  The evolutionary search will seek to optimize a tradeoff between model size and F1 score when determining the final student architecture.  For all experiments, we will run the evolutionary algorithm for 10 generations, with a population size of 30.  Further tuning of these parameters could be beneficial, but in practice it was observed that the algorithm seemed to converge by generation 10 in multiple experiments.

Now, in order to guarantee that the evolutionary search does not produce a student that is larger than the original teacher (which is possible due to the overparameterization of the supernet), a hard parameter limit is specified at the outset, roughly equivalent to the size of the original teacher model (25M parameters for the ResNet50, 11M for ResNet18, 5M for ResNet10t).

### 3.2 Assessing Fitness

As mentioned, the fitness function used to assess our population is a tradeoff between F1 score and model size.  We want to maximize the F1 score while penalizing model size to encourage a smaller final student architecture.  Our problem can thus be understood as **Multi-Objective Optimization**.  

For each member of the generation, we can specify the process as follows:

1. Check its size (number of parameters).  If this member exceeds the specified parameter limit, it is assigned a fitness of zero.  
2. Push some training data through the architecture to update the batch normalization statistics for that particular configuration
    - This helps us to not underestimate the performance of smaller models that are just “borrowing” BN statistics leftover from larger configurations.
    - We do not update weights here!  We just allow the BN statistics to tune to the specific architecture for a few batches of training data.
3. Evaluate Macro F1 score over 70 batches of validation data.
4. Add in an efficiency bonus for models that are smaller than the parameter constraint
    - This helps us incentivize smaller models that have roughly equivalent F1’s compared to larger configurations.
    - This efficiency bonus is scaled by a tuning parameter $\lambda$, set to 0.05 in our experiments.

Specifying this optimization problem mathematically, we want to **determine an optimal architectural configuration**, $\alpha^*$, within our search space $\mathcal{A}$.

$$
\alpha^*=\underset{\alpha \isin \mathcal{A}}{\arg\max}f(\alpha)
$$

$f(\alpha)$ is our fitness function, specified as:

$$
f(\alpha)=F_1 ( \alpha,W_{super})+\lambda\Bigl( 1-\frac{M(\alpha)}{C}\Bigr)
$$

Where $F_1(\alpha,W_{super})$ denotes our macro F1 score using our shared weights, $M(\alpha)$ denotes the parameter count of our subconfiguration, and $C$ denoted our parameter limit.

### 3.3 Reproduction

Following each iteration, the top 20% of the population are allowed to reproduce.  Crossover and Mutation combine to produce the next generation.

**Crossover**

For the two global genes, resolution and width, we randomly select this from either Parent A or Parent B.  

For the sequential genes that can differ at each stage of the architecture, depth and expansion, uniform crossover is done for each stage.  Each stage has a 50% chance of taking the depth from Parent A or Parent B.

**Mutation**

This is the exploration component, aiming to shake the algorithm out of local minima.  Mutation has a 20% probability of occurring.  If it occurs, a random choice is made for which gene to flip (input resolution, width, depth, or expansion ratio in the case of ResNet50).  Once the gene to flip is decided, the mutation simply changes the value to another valid option from the search space.

---

## IV. Results

While most of the implementation results will be left in the Jupyter notebook, here are some notes on the training and results of the supernet models.

Training Parameters

The same hyperparameters (excluding alpha and temperature for the distillation loss) were used for the training of all of our supernets.  This is identified as a weakness in the methodology - though similar in structure, the different supernet architectures likely have different optimal hyperparameters.  Due to time constraints on the project, this is what was necessary.  In future work, further hyperparameter tuning for each individual model would be highly beneficial.

- All models are trained over 60 epochs, the first 15 of which serve as warmup epochs where uniform path sampling is done.  After the warmup, GreedyNAS begins filtering the sampled paths.
- SGD is used for the optimizer, with small weight decay.
- Cosine Annealing is used on the LR, which has a maximum of 0.01.
- Validation is performed every 5 epochs, since it is fairly computationally slow due to the need for retuning Batch Normalization layers.

### **4.1 ResNet50**

The RN50 supernet features 191,075,391 parameters, far exceeding the original ResNet50 size.  Training the supernet took around 3.5 hours.  The evolutionary algorithm took 50 minutes to 1 hour to complete its search.  Fine-tuning took a little more than 1 hour over 30 epochs.

Training notes:

- The potentially dangerous path inversion described previously was avoided - in validation epochs, we clearly see that the MAX path is outperforming the MIN path as it ought to.
- The large difference between training F1 and validation F1 is concerning, as it may indicate overfitting.  However, with validation accuracy and F1 still gradually increasing, the training was allowed to continue.
- The soft loss remains fairly high.  I suspect this is due to noisy predictions from the teacher model.  With an F1 of only 0.49, the teacher is not the most reliable teacher.  A middling alpha of 0.5 was used to try to balance this - the supernet listened just as much to the hard loss as the soft loss.  However, we still see potential impacts on the supernet training from the weak teacher.

Results:

The optimal configuration determined by the NAS had ____ parameters, with the following configuration:

- Input resolution: 160x160
- Width: 1.0
- Depth: [2, 4, 2, 3]
- Expansion Ratio: [3, 2, 2, 3]

This indicates a highly expressive second stage (with four convolutional and normalization blocks) and more aggressive bottleneck compression at the first and last stages.  Input resolution is decreased from the default 224x224 to 160x160.  

This final student architecture achieved a validation F1 of 0.3802 during fine-tuning, and a final test F1 of 

### **4.2 ResNet18**

The RN18 supernet features 25,115,141 parameters.  Training took around 1 hour 10 minutes.  The evolutionary algorithm ran for a little less than 1 hour.  Fine-tuning took approximately 45 minutes.

Training Notes

- An alpha of only 0.4 was used, as the teacher is even more middling than the previous one, with a baseline F1 of 0.41.  This further decreased alpha encourages the supernet to pay more attention to the hard labels from the dataset than the teacher’s predictions.
- Path inversion was again successfully avoided - the MAX path is significantly better than the MIN path.

Results

The NAS determined an optimal configuration that consisted of ____ parameters, with the following configuration:

- Input resolution: 128x128
- Width: 0.75
- Depth: [3, 2, 1, 2]

This configuration favors fewer convolutional filters across the network (with a width of only 0.75), and a highly expressive initial stage (with an initial depth of 3).

This final student configuration achieved a validation F1 of 0.3891, and **a final test F1 of 0.36.**

### **4.3 ResNet10t**

The RN10t supernet features 16,129,571 parameters.  Training took around 1 hour.  The evolutionary algorithm ran for ~45 minutes.  Fine-tuning took only 30 minutes.

Training Notes

- An alpha of only 0.2 was used, as the teacher ResNet10t only has an F1 of 0.29.  We want some of the teacher’s knowledge, but we want to weight the hard labels much higher to allow the student a chance to surpass the teacher.
- Path inversion was again successfully avoided - the MAX path is significantly better than the MIN path.

Results

The NAS determined an optimal configuration that consisted of 2,819,693 parameters, with the following configuration:

- Input resolution: 128x128
- Width: 0.75
- Depth: [3, 2, 1, 2]

This configuration favors fewer convolutional filters across the network (with a width of only 0.75), and a highly expressive initial stage (with an initial depth of 3).

This final student configuration achieved a validation F1 of 0.3891, and **a final test F1 of 0.31.**

The performance of students to teachers can thus be shown by the following table, demonstrating the reduction in model parameters and the F1 scores.

![image.png](attachment:ff38d8d1-0153-4b6d-881f-29b94253c85b:image.png)

---
