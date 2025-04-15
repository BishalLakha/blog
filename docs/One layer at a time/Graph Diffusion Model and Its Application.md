
## 1 Introduction
The phenomenal performance of AI systems for image and video generation such as DALLE [^1], Stable Diffusion [^2], and Sora [^3] has captured the imagination of people and triggered the interest of many [^4], [^5],[^6] . Underlying these AI systems is a deep learning algorithm called diffusion models or denoising diffusion probabilistic models (DDPM). The diffusion model, first proposed by Sohl-Dickstein et al. [^7], and later improved by Ho et al. [^8], enabling practical use cases, is a physics-inspired generative model. In physics, diffusion refers to the process by which particles move from regions of higher concentration to regions of lower concentration to reach an equilibrium, mathematically captured by Fick’s laws [^9]. These laws describe the rate of diffusion by taking into account the concentration gradient between two points [^10]. High-dimensional data also behaves similarly to the randomly moving particles as they seek an optimal distribution or representation, making diffusion suitable for generative tasks [^10].

The diffusion model involves two main processes: the forward process (diffusion) and the reverse process (denoising), as illustrated in Fig. 1. The forward process requires gradually degrading the data, such as an image, through a multi-step noise application that converts it into a sample from a Gaussian distribution, discussed in detail in Section 1.1. Conversely, the reverse process, detailed in Section 1.2, involves training a deep neural network to reverse the noising steps, enabling the generation of new data from Gaussian-distributed samples [^11]. Unlike other generative models like Generative Adversarial Networks (GAN) [^12], diffusion models are easy to train and can scale well on parallel hardware, making them quite suitable for large-scale datasets [^11]. They also avoid the problem of instability during training and generate better results in comparison to those algorithms, leading to their increased adoption in research and applications [^13].

### 1.1 Forward Process
The forward process incrementally introduces noise into the data, transforming a clean data point $x_0$ into a series of increasingly noisy latent variables $x_1, x_2, \ldots, x_T$ through a Markov chain defined as:

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

Here, $\beta_t$ modulates the noise level, and $\mathcal{N}$ denotes a Gaussian distribution. The entirety of the forward process can be expressed as:

$$
q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})
$$

A notable aspect of this process is the direct sampling of $x_t$ at any noise level using:

$$
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I)
$$

where:

$$
\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)
$$

---

#### Algorithm 1: Training a DDPM [^15]

1. For every image $x_0$ in your training dataset:
2. **Repeat**:
   - Pick a random time step $t \sim \text{Uniform}[1, T]$.
   - Draw a sample $x_t \sim \mathcal{N}(x_t|\sqrt{\alpha_t}x_0, (1-\alpha_t)I)$, i.e.,
     $$
     x_t = \alpha_t x_0 + \sqrt{1-\alpha_t}z, \, z \sim \mathcal{N}(0, I)
     $$
   - Take a gradient descent step on:
     $$
     \nabla_\phi || \hat{x}_\phi(x_t) - x_0 ||^2
     $$
3. **Until convergence**

You can do this in batches, just like how you train any other neural network. Note that, here, you are training one denoising network $\hat{x}_\phi$ for all noisy conditions.

---
### 1.2 Reverse Process

The goal of the reverse process of the DDPM is to reconstruct the clean data by denoising, predicting $x_{t-1}$ from $x_t$ at each step using the equation:

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

where $\mu_\theta(x_t, t)$ and  $\Sigma_\theta(x_t, t)$ are functions modeled by a neural network $\theta$, determining the mean and covariance of the Gaussian distribution at each reverse step. The entire reverse process can be described as:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=T}^1 p_\theta(x_{t-1}|x_t)
$$

This process begins with the assumption that the final noisy data point, $x_T$, follows a Gaussian distribution, typically centered around zero with identity covariance. The subsequent denoising steps iteratively estimate the less noisy preceding states until the original data point $x_0$ is recovered.

---

##### Algorithm 2: Inference on a DDPM [^15]

1. You give a white noise vector $x_T \sim \mathcal{N}(0, I)$.
2. **For** $t = T, T-1, \ldots, 1$ **do**:
   1. Calculate $\hat{x}_\theta(x_t)$ using our trained denoiser.
   2. Update according to:
      $$
      x_{t-1} = \frac{(1-\alpha_{t-1})\sqrt{\alpha_t}}{1-\alpha_t} x_t + \frac{(1-\alpha_t)\sqrt{\alpha_{t-1}}}{1-\alpha_t} \hat{x}_\theta(x_t) + \sigma_q(t)z, \, z \sim \mathcal{N}(0, I)
      $$
      
3. **End For**

---
### 1.3 Training Objective

The training objective for the diffusion model is to maximize the variational lower bound (ELBO) on the log-likelihood expressed as:


$$
\text{ELBO}_{\phi, \theta}(x) = \mathbb{E}_{q_\phi(x_{1:T}|x_0)} [\log p_\theta(x_0|x_1)] 
- \text{D}_{\text{KL}}(q_\phi(x_T|x_0) || p(x_T)) 
- \sum_{t=2}^T \mathbb{E}_{q_\phi(x_t|x_0)} \left[ \text{D}_{\text{KL}} \left( q_\phi(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t) \right) \right]
$$

The first component of the equation, the reconstruction term, measures how well the model $p_\theta$ can reconstruct the initial data point $x_0$ from the latent variable $x_1$, using the log-likelihood $p_\theta(x_0|x_1)$. The second component involves the KL divergence, which measures the difference between the distribution $q_\phi(x_T|x_0)$ and the prior distribution $p(x_T)$. The last component, the consistency term, sums the KL divergences across transitions for $t = 2$ to $T$, measuring the alignment between the forward transition modeled by $q_\phi(x_{t-1}|x_t, x_0)$ and the reverse transition $p_\theta(x_{t-1}|x_t)$. The ELBO can be further simplified to get a loss function (see [^15]):

$$
\theta^* = \underset{\theta}{\text{arg min}} \sum_{t=1}^T \frac{1}{2\sigma^2(t)} 
\frac{(1-\alpha_t)^2 \alpha_{t-1}}{(1-\alpha_t)^2} \mathbb{E}_{q(x_t|x_0)} \left[ \| \hat{x}_\theta(x_t) - x_0 \|^2 \right].
$$

Algorithm 1 and 2 summarize the training and inference procedures for the diffusion model.


### 1.4 Guided Diffusion

In many applications, like text prompt-based image generation, generating samples from a conditional distribution \( p(x|c) \) is desired where \( c \) is the conditioning variable. But the diffusion model can assign insufficient weight on these conditions, so additional pressure called "guidance" is applied, resulting in guided diffusion [^11]. 

There are primarily two types of guidance: **classifier guidance** and **classifier-free guidance**.
1. **Classifier Guidance**: In this approach, a separate classifier model $p(c|x)$is utilized along with a diffusion model to drive the sample generation towards desired characteristics defined by the conditional label $c$ [^16].
2. **Classifier-Free Guidance**: In this approach, rather than employing a separate classifier for guidance, conditional and unconditional diffusion processes are jointly trained to achieve the desired outcomes [^17].
---
## 2. Graph Diffusion Models

A graph $G$, defined as a pair $G = (V, E)$, is a fundamental structure in mathematics and computer science used to model relationships and interaction between the objects. Here $V$ is a set of vertices and $E$ is a set of edges, each connecting a pair of vertices. Graphs can be used to model different complex and interconnected data like social networks, recommendation systems, biological networks, and other areas, which makes adaptation of generative models like diffusion models for graphs quite crucial.

As illustrated in Fig 2, DDPM on graphs involves forward and reverse processes where the primary focus is on creating a transition kernel for the Markov chain [^18]. Various methods have been proposed to achieve that. Haefeli et al. [^19] proposed using discrete noise for the forward Markov process instead of using continuous Gaussian perturbations, which ensured that for every intermediate step, the graph remained discrete, resulting in better and faster sample generation. We discuss the forward and reverse process proposed by [^19] in Section 2.1 and 2.2.

### 2.1 Forward Process

The diffusion model generates a sequence from an initial simple graph $A_0$ (adjacency matrix representing the graph) to white noise $A_T$ through a series of increasingly noisier graphs $A_1, A_2, \ldots, A_T$. Both $A_0$ and $A_T$ are adjacency matrices where $A_0$ is a sample from the dataset and $A_T$ is an Erdős–Rényi [^20] random graph.

Each element $a_{ij}^t$ of the adjacency matrix between nodes $i$ and $j$ is encoded as a one-hot vector and transformed by a double stochastic matrix $Q_t$ defined as:

$$
Q_t = 
\begin{bmatrix}
1 - \beta_t & \beta_t \\
\beta_t & 1 - \beta_t
\end{bmatrix}
$$

Here, $\beta_t$ indicates the probability of the edge state not changing. This formulation allows for direct sampling at any timestep $t$ independently for each edge and non-edge, facilitating the simplification of the sampling process without relying on previous timesteps.

### 2.2 Reverse Process

Reverse process aims to recover the original graph from the noise. The reverse transition is denoted as $q(A_{t-1}|A_t, A_0)$ and is crucial for training to learn to denoise the graphs. The reverse transition probabilities are derived from the forward probabilities with a dependence on the initial graph \( A_0 \) to ensure accurate regeneration of the graph given as:

$$
q(A_{t-1}|A_t, A_0) = \frac{q(A_t | A_{t-1}) q(A_{t-1} | A_0)}{q(A_t | A_0)}.
$$



### 2.3 Latent Diffusion

Diffusion over discrete graph space can suffer from different issues, primarily high modeling complexity, complex relational information leading to limited semantic learning, and consequently poor performance [^21]. Instead, using a latent space can improve efficiency with faster sampling and produce better samples by producing smoother representation [^21],[^22], [^23]. 

This is generally achieved by first training a variational graph autoencoder (VGAE) to capture topological information and then applying diffusion with some conditioning on its latent space to enhance the representation, and finally using the decoder to generate the graph [^21], [^22],[^23], [^24].

---
## 3 Applications

Graph diffusion models have demonstrated substantial efficacy across a diverse range of fields like biology, chemistry, physics, computer science, and others. In this section, we will explore several applications of these models, highlighting their role and impact in research and industry.
### 3.1 Molecule and Protein Design

Diffusion models can generate novel molecules by learning the distribution of existing molecular graphs. This is particularly useful for discovering new drugs with desired properties. For example, **DiffHopp** [^25] is a graph diffusion model tailored for scaffold hopping in drug design, which modifies the core structure of known active compounds to generate new chemical entities while preserving essential molecular features. Similarly, they are powerful tools for predicting protein structures and interactions, which is vital to understanding biological processes and designing new therapeutics. For example, models such as **DiffDock** [^26] facilitate molecular coupling by predicting how small molecules bind to proteins.

### 3.2 Materials Design

Graph diffusion models are used to design new materials with specific properties by generating graphs that represent the structures of the material, which can then be analyzed for their physical and chemical properties. By learning the graph structures of existing materials, these models can propose new materials with enhanced characteristics, such as increased strength or conductivity, which makes the methods quite relevant in fields like nanotechnology and materials science [^27].
### 3.3 Combinatorial Optimization

Graph diffusion models are also applied in solving combinatorial optimization problems, where the goal is to find the best solution from a finite set of possible solutions. Models like **DIFUSCO** [^28] can generate candidate solutions for problems like the traveling salesman and maximal independent set problem.
### 3.4 xAI

Machine learning models are generally "black box" in nature, making them untrustworthy and unreliable. Explainable AI (xAI) refers to methods and systems that make such models explainable and interpretable. Graph Neural Networks (GNNs) are also black-box in nature, and diffusion models like **D4Explainer** [^29] can produce both counterfactual and model-level explanations for GNNs.

---
## References
[^1]:
    Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022.

[^2]:
    Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2021.

[^3]:
    Video generation models as world simulators — OpenAI.

[^4]:
    Thomas Macaulay. OpenAI’s new image generator sparks both excitement and fear, April 2022.

[^5]:
    Kevin Roose. AI-generated art is already transforming creative work. *The New York Times*, October 2022.

[^6]:
    OpenAI reveals Sora, a tool to make instant videos from written prompts, February 2024.

[^7]:
    Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. *(arXiv:1503.03585)*, November 2015. [arXiv:1503.03585 [cond-mat, q-bio, stat]].

[^8]:
    Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In *Advances in Neural Information Processing Systems*, volume 33, page 6840–6851. Curran Associates, Inc., 2020.

[^9]: Fick’s laws of diffusion, July 2024. Page Version ID: 1235588853.

[10]: Diffusion models.

[^11]:
    Christopher M. Bishop and Hugh Bishop. *Deep Learning: Foundations and Concepts*. 
Springer International Publishing, Cham, 2024.

[^12]:
    Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks, 2014.

[^13]:
    Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. *(arXiv:2209.00796)*, June 2024. [arXiv:2209.00796 [cs]].

[^14]:
    Image generation with diffusion models using Keras and TensorFlow — by Vedant Jumle — Towards Data Science.

[^15]:
    Stanley H. Chan. Tutorial on diffusion models for imaging and vision. *(arXiv:2403.18103)*, March 2024. [arXiv:2403.18103 [cs]].

[^16]:
    Prafulla Dhariwal and Alex Nichol. Diffusion models beat GANs on image synthesis, 2021.

[^17]:
    Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*, 2021.

[^18]:
    Chengyi Liu, Wenqi Fan, Yunqing Liu, Jiatong Li, Hang Li, Hui Liu, Jiliang Tang, and Qing Li. Generative diffusion models on graphs: Methods and applications. *(arXiv:2302.02591)*, August 2023. [arXiv:2302.02591 [cs]].

[^19]:
    Kilian Konstantin Haefeli, Karolis Martinkus, Nathanaël Perraudin, and Roger Wattenhofer. Diffusion models for graphs benefit from discrete state spaces, 2023.

[^20]:
    Paul Erdős and Alfréd Rényi. On the evolution of random graphs. *Publ. Math. Inst. Hungar. Acad. Sci.*, 5:17–61, 1960.

[^21]:
    Ling Yang, Zhilin Huang, Zhilong Zhang, Zhongyi Liu, Shenda Hong, Wentao Zhang, Wenming Yang, Bin Cui, and Luxia Zhang. Graphusion: Latent diffusion for graph generation. *IEEE Transactions on Knowledge and Data Engineering*, page 1–12, 2024.

[^22]:
    Iakovos Evdaimon, Giannis Nikolentzos, Michail Chatzianastasis, Hadi Abdine, and Michalis Vazirgiannis. Neural graph generator: Feature-conditioned graph generation using latent diffusion models. *arXiv preprint arXiv:2403.01555*, 2024.

[^23]:
    Minkai Xu, Alexander S Powers, Ron O Dror, Stefano Ermon, and Jure Leskovec. Geometric latent diffusion models for 3d molecule generation. In *International Conference on Machine Learning*, pages 38592–38610. PMLR, 2023.

[^24]:
    Cong Fu, Keqiang Yan, Limei Wang, Tao Komikado, Koji Maruhashi, Kanji Uchino, Xiaoning Qian, and Shuiwang Ji. A latent diffusion model for protein structure generation.

[^25]:
    Jos Torge, Charles Harris, Simon V. Mathis, and Pietro Lio. Diffhopp: A graph diffusion model for novel drug design via scaffold hopping, 2023.

[^26]:
    Gabriele Corso, Hannes Stärk, Bowen Jing, Regina Barzilay, and Tommi Jaakkola. Diffdock: Diffusion steps, twists, and turns for molecular docking, 2023.

[^27]:
    Mengchun Zhang, Maryam Qamar, Taegoo Kang, Yuna Jung, Chenshuang Zhang, Sung-Ho Bae, and Chaoning Zhang. A survey on graph diffusion models: Generative AI in science for molecule, protein and material. *arXiv preprint arXiv:2304.01565*, 2023.

[^28]:
    Zhiqing Sun and Yiming Yang. Diffusco: Graph-based diffusion solvers for combinatorial optimization. *Advances in Neural Information Processing Systems*, 36:3706–3731, 2023.

[^29]:
    Jialin Chen, Shirley Wu, Abhijit Gupta, and Rex Ying. D4explainer: In-distribution explanations of graph neural network via discrete denoising diffusion. *Advances in Neural Information Processing Systems*, 36, 2024.
