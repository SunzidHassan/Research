# LLM-OSL
## Table of Contents

- [LLM-OSL](#llm-osl)
  - [Table of Contents](#table-of-contents)
  - [To-Do](#to-do)
  - [OSL](#osl)
    - [Semantic OSL](#semantic-osl)
    - [Probabilistic Inference in OSL](#probabilistic-inference-in-osl)
  - [Multimodal LLM-based Navigation](#multimodal-llm-based-navigation)
    - [Semantic Graph Representation from Scene](#semantic-graph-representation-from-scene)
    - [ObjectNav Challenge](#objectnav-challenge)
    - [3D Semantic Map from Scene](#3d-semantic-map-from-scene)
    - [Vector Database Representation from Scene](#vector-database-representation-from-scene)
  - [Simulation Platforms](#simulation-platforms)
  - [Neural Fields in Robotics](#neural-fields-in-robotics)
    - [NeRF in Robotics Survey](#nerf-in-robotics-survey)
    - [Planning/Navigation](#planningnavigation)

---

Format:

* **Short Title**: "TITLE", CONFERENCE, YEAR. [[Paper](link)] [[Code](link)] [[Website](link)]

---

## To-Do

Literature review:

* Infotaxis on:Anemotactic OSL method to generate 2D map
* VLM to generate semantic 2D map

Issues:

* No sim with multiple gasses
* No sim with e-nose?

Tools:

* Isaac visual SLAM

Olfaction processing:

* Find gas sensors (Adafruit, e-nose, etc.), and simulators.
* Filter gasses that can be sensed from [Flavornet](https://www.flavornet.org/) to get a lookup table of gas to concept

Multimodal sensor fusion:

* Open-set 3D map
* Multimodal NeRF

---

## OSL

* **OSL Review**: "Recent Progress and Trend of Robot Odor Source Localization", IEEJ Transactions on Electrical and Electronic Engineering, 2021. [[Paper](https://onlinelibrary.wiley.com/doi/full/10.1002/tee.23364)]

---

### Semantic OSL
>
>[Flavornet](https://www.flavornet.org/)

* **Semantic OSL**: "A Semantic-Based Gas Source Localization with a Mobile Robot Combining Vision and Chemical Sensing", Sensors, 2018. [[Paper](https://www.mdpi.com/1424-8220/18/12/4174)]

>Monroy et al. `\cite{monroy2018semantic}` discussed using semantic vision and olfaction sensing for gas source localization. They used an electronic-nose to detect multiple gasses for semantic olfactory sensing. They utilized the YOLOv3 object detector to detect indoor objects for semantic vision sensing, and used a depth camera to map those object in a global map. Finally, they used bayesian probabilistic framework on ontological knowledge to assess probable sources of specific odor classes, and used MDP to plan path to those sources.

* **Semantic OSL for USaR**: "A low cost, mobile e-nose system with an effective user interface for real time victim localization and hazard detection in USaR operations", Measurement: Sensors, 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S2665917421000118)]

>Anyfantis et al. `\cite{anyfantis2021low}` proposed a e-nose system coupled with a remote controlled mobile robot to assist Urban Search and Rescue (USaR) operations. The e-nose system treats Ammonia gas as a potential human presence indicator, and CO, H2S and combustible gasses as hazard indicator.

* **Semantic Olfaction-Vision**: "Olfaction, Vision, and Semantics for Mobile Robots. Results of the IRO Project", Sensors, 2019. [[Paper](https://www.mdpi.com/1424-8220/19/16/3488)]

>Monroy et al. `\cite{monroy2019olfaction}` discussed the results of five-year long project IRO: Improvement of the sensory and autonomous capability of Robots through Olfaction. The project covers 3 main goals - (1) design of an e-nose, (2) object recognition using RGB-D vision system, and (3) exploiting high-level olfactory and visual semantic information in planning and executing.

---

### Probabilistic Inference in OSL

* **Infotaxis**: "'Infotaxis' as a strategy for searching without gradients", Nature, 2007. [[Paper](https://www.nature.com/articles/nature05464)]

>$\mathcal{T_{t}}$ time and coordinates of hit.
>$P_t(\mathbf{r_0})$ is the posterior probability distribution of unknown source $\mathbf{r_0}$.
>Odor particles are emitted by source at rate $R$, have finite lifetime $\tau$, propagate with effective diffusivity $D$ and are advected by a mean current or wind $\mathbf{V}$.
>Expected search time $<T>$ is bounded by $<T>\geq e^{S-1}$, where entropy $S\equiv-\int{\text{dx}P(\mathbf{x})}$. The entropy quantifies how spread-out the distribution is, and goes to zero when the position of the source is localized. The rate of acquisition of information is quantified by the rate of entropy reduction.  
>To balance exploration-exploitation, 'infotaxis' chooses the direction that locally maximises the expected rate of information acquisition, thereby maximises reduction of entropy of the posterior probability field.  
>If the searcher is at location $\mathbf{r}$ at time $t$ and gathered information into the field $P_t(\mathbf{r_0})$ having entropy $S$. The variation of entropy expected upon moving to one of the neighbouring points $\mathbf{r}_j$ (or standing still) is: $\overline{\Delta S}(\mathbf{r_i} \to \mathbf{r_j}) = P_t[\mathbf{r_j}](-S) + \left[1 - P_t(\mathbf{r_j})\right] \left[\rho_0(\mathbf{r_j}) \Delta S_0 + \rho_1(\mathbf{r_j}) \Delta S_1 + \dots \right]$
> The first term on the right-hand side corresponds to finding the source, that is, $P_{t+1}$ becoming a $\delta$-function and entropy becoming zero, which occurs with estimated probability $P_t(\mathbf{r}_j)$. The second term one the right-hand side corresponds to the alternative case when the source is not found at $\mathbf{r}_j$. Symbols $\rho_k(\mathbf{r}_j)$ denote the probability that $k$ detections be made at $\mathbf{r}_j$ during a time-step $\Delta t$, given by a Poisson law $\rho_k=\frac{h^ke^{-h}}{k!}$ for the independent detections. The expected number of hits is estimated as $h(\mathbf{r}_j)\equiv\Delta t\int P_t(\mathbf{r}_0)R(\mathbf{r}_j|\mathbf{r}_0)d\mathbf{r}_0$, with $R(\mathbf{r}|\mathbf{r}_0)$ denoting the mean rate of hits at position $\mathbf{r}$ if the source is located in $\mathbf{r}_0$. The symbols $\Delta S_k$ denote the change of entropy between the fields $P_{t+1}(\mathbf{r}_0)$ and $P_{t}(\mathbf{r}_0)$.
>Two effects contribute to $\Delta S_k$: first $P_{t+1}(\mathbf{r}_0)\equiv0$ because the source was not found; and second, the estimated prosterior probabilities are modified by the $k$ cues received. The first term on the right-hand side of the equation is the exploitative term, weighing only the event that the source is found at the point $\mathbf{r}_j$ and favoring motion to maximum likelihood points. The second term on the right-hand side of the equation is the information gain from receiving additional cues.

* **OTTO**: "A Python package to simulate, solve and visualize the source-tracking POMDP", Journal of Open Source Software, 2022. [[Paper](https://royalsocietypublishing.org/doi/10.1098/rspa.2022.0118)] [[Code](https://github.com/C0PEP0D/otto)]

* **Probabilistic MAP OSL VGR**: "Robotic Gas Source Localization With Probabilistic Mapping and Online Dispersion Simulation", CONFERENCE, YEAR. [[Paper](https://arxiv.org/abs/2304.08879)]


---
## Multimodal LLM-based Navigation

### Semantic Graph Representation from Scene
* **ConceptGraphs**: "Open-Vocabulary 3D Scene Graphs for Perception and Planning", ICRA, 2024. [![Paper](https://img.shields.io/badge/arXiv24-b22222)](http://arxiv.org/abs/2309.16650) [![Star](https://img.shields.io/github/stars/concept-graphs/concept-graphs.svg?style=social&label=Star)](https://github.com/concept-graphs/concept-graphs) [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://concept-graphs.github.io/)
>**ConceptGraphs** builds open vocabulary 3D scene graphs that can be queried with natural language. LLM and VLM is used to extract image features and captions, which are then used to build an open-vocab 3D scene graph.  
>**Querying ConceptGraph**: *Query using VLM*: taking CLIP feature vectors of the graph objects and query object, and taking their cosine similarity. *Query using LLM*: serialized scene graph is fed to LLM using simple JSON format. LLM uses object captions to select the most appropriate object for the query. This works better for complex queries.  
>**Methodology**: Given an input sequence of RGB-D images with poses, ConceptGraph uses openvacab instance segmentation models to get 2D masks for the objects in those images. Depth information of the masks is used to project individual point clouds of the objects in 3D. The image crops are again fed to VLMs like CLIP to get semantic feature vectors for the objects. The semantic pointcloud and semantic feature vectors are used to measure the spatial and semantic similarity of the objects to existing objects in the map - each objects are merged into one, or added anew. Repeating this process generates an object-based 3D map from input images. Once the map is built, all the nodes is received for the scene graph. An LLM is used to generate edges among the object nodes - LLM will generate positional edge (e.g., stool on top of carpet). , grouping them over time into potential 3D objects. 

### ObjectNav Challenge
* **VLFM**: "Vision-Language Frontier Maps for Zero-Shot Semantic Navigation". [![Paper](https://img.shields.io/badge/ICRA24-8A2BE2)]() [![Paper](https://img.shields.io/badge/arXiv24-b22222)](https://arxiv.org/abs/2312.03275) [![Star](https://img.shields.io/github/stars/bdaiinstitute/vlfm.svg?style=social&label=Star)](https://github.com/concept-graphs/concept-graphs) [![Project_Page](https://img.shields.io/badge/Project_Page-00CED1)](https://naoki.io/portfolio/vlfm)

>**Introduction:**  
>Yokoyama et al. proposed VLFM - a zero-shot semantic navigation using frontier-based exploration.
>Framework: VLFM builds occupancy maps from depth observations to identify frontiers, and language-grounded value map using RGB and VLM. This map is used to identify the most promising frontier to explore for finding instance of a given target object category.
>Validation: Habitat simulator - Gibson, Habitat-Matterport 3D, Matterport 3D.
>
>**Related Works:**
>Zero-shot ObjectNav
>
> * Frontier-based exploration involvs visiting boundaries between explored and unexplored areas on a map that is iteratively built with exploration. Froentiers are chosen based on infotaxis, nearest-first (CoW), LLM selection (LGX, ESC). SemUtil use BERT to embed class labels of objects near frontier, and compare them to text embeding of target object to select frontier.  
>
>**Problem Formulation: ObjectNav**
>
>* finding a target object category in unknown environment.
>* Access to RGB-D camera, odometry.
>* Actions: forward, left, right, look up, look down, stop.
>* Success: if stop is called within 1m of any instance of target object in 500/fewer steps.
>
>**Vision-Language Frontier Maps:**
>
>1. Initialization: robot rotates to set up frontier and value maps.
>2. Exploration (till target is detected): update frontier and value maps, generate frontier waypoints and select most valueable waypoint to navigate to the target object.
>3. Goal navigation: if target object is detected, it navigates to the nearest point and triggers if it's within sufficient proximity.
>
>* Waypoint generation: obstacle map from depth and odometry observations (like SLAM). Identify each boundary that separates explored and unexplored areas, identifying the midpoints as potential frontier waypoints. Quantify of waypoints will vary until the entire environment has been explored (unsuccessfull STOP).
>* Value map generation: similar to obstacle map - with value and confidence scores.
>   * Value score: BLIP-2 cosine similarity score from RGB observation and text prompt containing the target object ("Seems like there is a `target object` ahead"). These scores are then projected onto value channel of the top-down value map.
>   * Confidence score: how to update semantic pixel value. If pixel not seen, don't update value. Pixels along 0-degree FOV has confidence value of 1, left and right edge has 0. In between values follow $\cos^2\left(\frac{\theta}{\theta_{\text{fov}/2}}\times\pi/2\right)$, wehre $\theta$ is the angel between pixel and optical axis, and $\theta_{\text{fov}}$ is the horizontal FOV of the robot's camera. If a pixel is seen twice, an confidence weighted average is taken.
>* Ojbect detection: if object is detected using YOLOv7, Mobile-SAM is used to extract its contour using RGB image and bounding box. The contour is matched with depth image to determine the closest point, which is then used as the goal waypoint.
>* Waypoint navigation: the robot has either frontier or target object waypoint. Variable Experience Rollout (VER) RL method is used to train Point Goal Navigation (PointNav) policy, which is used to determine action to reach current waypoint using visual observation and odometry.
>
> Baselines:
>
>* CoW: explores closest frontier until target object is detected using CLIP, then direct navigation to the object.
>* ESC, SemUtil: performs semantic frontier-based nav, frontiers are evaluated using detected object>LLM.
>* ZSON: trained on ImageNav.
>* PONI, SemExp: build maps during navigation, train task specific poligies to perform semantic inference of likely target location.
>* PIRLNav: trained form human demonstrations.
>* RegQLearn: RL

* **NaVid**: "Video-based VLM Plans the Next Step for Vision-and-Language Navigation", RSS , 2024. [[Paper](https://arxiv.org/pdf/2402.15852.pdf)] [[Code](https://github.com/jzhzhang/NaVid-VLN-CE)] [[Website](https://pku-epic.github.io/NaVid/)]

* **OVSG**: "Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs", CoRL, 2023. [[Paper](https://arxiv.org/abs/2309.15940)] [[Code](https://github.com/changhaonan/OVSG)] [[Website](https://ovsg-l.github.io/)]

* **VLMaps**: "Visual Language Maps for Robot Navigation", CONFERENCE, YEAR. [[Paper](https://arxiv.org/pdf/2210.05714.pdf)] [[Code](https://github.com/vlmaps/vlmaps.git)] [[Website](https://vlmaps.github.io/)] [[CoLab](https://colab.research.google.com/drive/1xsH9Gr_O36sBZaoPNq1SmqgOOF12spV0?usp=sharing)] [[Blog](https://ai.googleblog.com/2023/03/visual-language-maps-for-robot.html?m=1)]

### 3D Semantic Map from Scene
* **Kimera**: "Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization and Mapping", ICRA, 2020. [[Paper](https://arxiv.org/abs/1910.02490)] [[Code](https://github.com/MIT-SPARK/Kimera-Semantics?tab=readme-ov-file)]

>Kimera is an open-source library for real-time semantic visual SLAM. It allows mesh construction and semantic labelling in 3D.

* **ConceptFusion**: "ConceptFusion: Open-set Multimodal 3D Mapping", Robotics: Science and Systems (RSS), 2023. [[Paper](https://arxiv.org/abs/2302.07241)] [[Code](https://github.com/concept-fusion/concept-fusion)] [[Website](https://concept-fusion.github.io/)]

>Jatavallabhula et al. `\cite{jatavallabhula2023conceptfusion}` proposed ConceptFusion - a pixel-alighed open-set 3D mapping using foundation. The three objectives of ConceptFusion are (i) open-set (classifying data to various concepts and levels of details) and (ii) multimodal (language, image, audio, 3D geometry) using LLM and are zero-shot queryable by text, image, audio and quick queries. The system was tested on simulated and real-world datasets, tabletop manipulation, autonomous driving platform.
>
>**Methdology**:
>
>* Given sequence of observations $\mathcal{I}$ an open-set multimodal 3D map $\mathcal{M}$ is built. A foundation model $\mathcal{F}_{mode}$ is used as modality-specific encoder to convert query concept of specific mode (text, audio, image...) to query vector $q_{mode}$.
>
>* Fusing pixel-aligned foundation features to 3D
>Map representation: 3D map $\mathcal{M}$ is an unordered set of points, each with a vertex position, normal vector, confidence count, color vector, concept vector.
>...

### Vector Database Representation from Scene
* **ReMEmbR**: "ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation", arXiv, 2024. [[Paper](https://arxiv.org/abs/2409.13682)] [[Code](https://github.com/NVIDIA-AI-IOT/remembr)] [[Website](https://nvidia-ai-iot.github.io/remembr/)] [[Blog](https://developer.nvidia.com/blog/using-generative-ai-to-enable-robots-to-reason-and-act-with-remembr/)]
>Anwar et al. `\cite{anwar2024remembr}` proposes ReMEmbR, that combines LLMs, VLMs and retrieval-augmented generation (RAG) to enable robots to reason and take actions over whey they observe during long-horizon deployment. It uses VLMs and vector databases to build long-horizon semantic memory. An LLM is used to query and reason over the memory. The system is tested on NVIDIA Jetson Orin edge computing devices on a mobile robot.

---

## Simulation Platforms

* **Short Title**: "TITLE", CONFERENCE, YEAR. [[Paper](link)] [[Code](link)] [[Website](link)]

* **GSL-Bench**: "GSL-Bench: High Fidelity Gas Source Localization Benchmarking Tool", ICRA, 2024. [[Paper](https://ieeexplore.ieee.org/document/10610755)] [[Code](https://github.com/Herwich012/GSL-Bench)] [[Website](https://sites.google.com/view/gslbench/)]

        Erwich et al. `\cite{erwich2024gsl}` proposed a Gas Source Localization benchmarking tool using NVIDIA's **Isaac Sim** simulation platform with **OpenFOAM** computational fluid dynamics software. They tested E. Coli, dung beetle and random walker algorithms in six indoor warehouse settings to test their platform.

* **Fleet UAV OSL Simulation**: "Odor source localization in outdoor building environments through distributed cooperative control of a fleetof UAVs", Expert Systems with Applications, 2024. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417424001970)]

        Jabeen et al. `\cite{jabeen2024odor}` used **ROS-Gazebo** simulation environment with **GADEN** odor simulator for simulating a fleet of UAVs for 3-D OSL in urban outdoor environment. Their proposed navigation algorithm used particle swarm optimization (PSO) for generating waypoints from the information gathered by the UAV fleet, and Estremum Seeking Control (ESC) for guiding the UAVs towards the waypoints to localize the odor source.

* **Infotaxis-PF UAV 3D-GSL**: "Gas Source Localization for UAV Using Infotaxis-Based Three-Dimensional Algorithm and Particle Filtering", Chinese Control Conference (CCC), 2024. [[Paper](https://ieeexplore.ieee.org/document/10662172)] [[Code](link)] [[Website](link)]

        He et al. `\cite{he2024gas}` used **ROS-Gazebo** simulation platform for OSL using UAV in indoor environments. They used combination of Infotaxis and Particle Filtering as the OSL algorithm. 

* **RL UAV GSL**: "Gas concentration mapping and source localization for environmental monitoring through unmanned aerial systems using model-free reinforcement learning agents", Plos one, 2024. [[Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296969)]

        Husnain et al. `\cite{husnain2024gas}` used **ROS-Gazebo** simulation environment to simulate grid environment for OSL with UAVs. They used an array of MOX sensors to identify the direction of gas plumes. They trained a Q-learning model for localizing plume sources.

* **Configurable PSL Simulation**: "Configurable simulation strategies for testing pollutant plume source localization algorithms using autonomous multisensor mobile robots", International Journal of Advanced Robotic Systems, 2022. [[Paper](https://journals.sagepub.com/doi/full/10.1177/17298806221081325)]

        Lewis et al. `\cite{lewis2022configurable}` used **ROS-Gazebo** simulation platform for plume source localization with multi-sensor mobile-robots, and used MATLAB and Simulink platform as dispersion model generator.

* **RL GSL**: "Robotic Information Gathering With Reinforcement Learning Assisted by Domain Knowledge: An Application to Gas Source Localization", IEEE Access, 2021. [[Paper](https://ieeexplore.ieee.org/document/9326418)]

        Wiedmann et al. `\cite{wiedemann2021robotic}` used **ROS-Gazebo** simulation platform to simulate outdoor forest environment for OSL with mobile-robot. They used domain knowledge assisted reinforcement learning method as the OSL algorithm.

---

## Neural Fields in Robotics

* **NeRF**: "Representing Scenes as Neural Radiance Fields for View Synthesis", Communications of the ACM, 2021. [[Paper](https://arxiv.org/abs/2003.08934)]

[Neural Radiance Fields (NeRF) Tutorial in 100 lines of PyTorch code](https://papers-100-lines.medium.com/neural-radiance-fields-nerf-tutorial-in-100-lines-of-pytorch-code-365ef2a1013)

---
### NeRF in Robotics Survey

* **NeRF-Robotics**: "Neural Fields in Robotics: A Survey", arXiv, 2024. [[Paper](https://arxiv.org/abs/2410.20220)] [[Repo](https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics?tab=readme-ov-file#slam)]

<!-- ---
### Object Pose Estimation

---
### SLAM

---
### Manipulation/RL

---
### Object Reconstruction

---
### Physics -->

---

### Planning/Navigation

* **NeRF-Navigation**: "Vision-Only Robot Navigation in a Neural Radiance World.", CONFERENCE, YEAR. [[Paper](https://mikh3x4.github.io/nerf-navigation/assets/NeRF_Navigation.pdf)] [[PyTorch Code](https://github.com/mikh3x4/nerf-navigation)] [[Website](https://mikh3x4.github.io/nerf-navigation/)]
