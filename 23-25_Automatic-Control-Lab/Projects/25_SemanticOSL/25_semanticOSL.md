# LLM-OSL
Format:
* ** Short Title**: "TITLE", CONFERENCE, YEAR. [[Paper](link)] [[Code](link)] [[Website](link)]

---
## To-Do
Olfaction processing:
- Find gas sensors (Adafruit, e-nose, etc.), and simulators.
- Filter gasses that can be sensed from [Flavornet](https://www.flavornet.org/) to get a lookup table of gas to concept

Vision processing:
- RGB-D > Semantics > Map/Memory

Multimodal sensor fusion:
- Open-set 3D map
- Multimodal NeRF

---
## Overview
* [OSL Review](#osl-review)
* [Semantic OSL](#semantic-osl)
* [Multimodal LLM-based Navigation](#multimodal-llm-based-navigation)
* [Simulation Platforms](#simulation-platforms)


---
## OSL Review
* **OSL Review**: "Recent Progress and Trend of Robot Odor Source Localization", IEEJ Transactions on Electrical and Electronic Engineering, 2021. [[Paper](https://onlinelibrary.wiley.com/doi/full/10.1002/tee.23364)]

---
## Semantic OSL
* ** Short Title**: "TITLE", CONFERENCE, YEAR. [[Paper](link)] [[Code](link)] [[Website](link)]

Notes:
- https://www.flavornet.org/

* **Semantic OSL**: "A Semantic-Based Gas Source Localization with a Mobile Robot Combining Vision and Chemical Sensing", Sensors, 2018. [[Paper](https://www.mdpi.com/1424-8220/18/12/4174)]

Monroy et al. `\cite{monroy2018semantic}` discussed used semantic vision and olfaction sensing for gas source localization. They used an electronic-nose to detect multiple gasses for semantic olfactory sensing. They utilized the YOLOv3 object detector to detect indoor objects for semantic vision sensing, and used a depth camera to map those object in a global map. Finally, they used bayesian probabilistic framework on ontological knowledge to assess probable sources of specific odor classes, and used MDP to plan path to those sources.

* **Semantic OSL for USaR**: "A low cost, mobile e-nose system with an effective user interface for real time victim localization and hazard detection in USaR operations", Measurement: Sensors, 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S2665917421000118)]

Anyfantis et al. `\cite{anyfantis2021low}` proposed a e-nose system coupled with a remote controlled mobile robot to assist Urban Search and Rescue (USaR) operations. The e-nose system treats Ammonia gas as a potential human presence indicator, and CO, H2S and combustible gasses as hazard indicator.


* **Semantic Olfaction-Vision**: "Olfaction, Vision, and Semantics for Mobile Robots. Results of the IRO Project", Sensors, 2019. [[Paper](https://www.mdpi.com/1424-8220/19/16/3488)]

Monroy et al. `\cite{monroy2019olfaction}` discussed the results of five-year long project IRO: Improvement of the sensory and autonomous capability of Robots through Olfaction. The project covers 3 main goals - (1) design of an e-nose, (2) object recognition using RGB-D vision system, and (3) exploiting high-level olfactory and visual semantic information in planning and executing.


---
## Multimodal LLM-based Navigation
* ** Short Title**: "TITLE", CONFERENCE, YEAR. [[Paper](link)] [[Code](link)] [[Website](link)]

* **ReMEmbR**: "ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation", arXiv, 2024. [[Paper](https://arxiv.org/abs/2409.13682)] [[Code](https://github.com/NVIDIA-AI-IOT/remembr)] [[Website](https://nvidia-ai-iot.github.io/remembr/)] [[Blog](https://developer.nvidia.com/blog/using-generative-ai-to-enable-robots-to-reason-and-act-with-remembr/)]

Anwar et al. `\cite{anwar2024remembr}` proposes ReMEmbR, that combines LLMs, VLMs and retrieval-augmented generation (RAG) to enable robots to reason and take actions over whey they observe during long-horizon deployment. It uses VLMs and vector databases to build long-horizon semantic memory. An LLM is used to query and reason over the memory. The system is tested on NVIDIA Jetson Orin edge computing devices on a mobile robot.


* **NaVid**: "Video-based VLM Plans the Next Step for Vision-and-Language Navigation", RSS , 2024. [[Paper](https://arxiv.org/pdf/2402.15852.pdf)] [[Code](https://github.com/jzhzhang/NaVid-VLN-CE)] [[Website](https://pku-epic.github.io/NaVid/)]


* **OVSG**: "Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs", CoRL, 2023. [[Paper](https://arxiv.org/abs/2309.15940)] [[Code](https://github.com/changhaonan/OVSG)] [[Website](https://ovsg-l.github.io/)]


* **VLMaps**: "Visual Language Maps for Robot Navigation", CONFERENCE, YEAR. [[Paper](https://arxiv.org/pdf/2210.05714.pdf)] [[Code](https://github.com/vlmaps/vlmaps.git)] [[Website](https://vlmaps.github.io/)] [[CoLab](https://colab.research.google.com/drive/1xsH9Gr_O36sBZaoPNq1SmqgOOF12spV0?usp=sharing)] [[Blog](https://ai.googleblog.com/2023/03/visual-language-maps-for-robot.html?m=1)]


* **Kimera**: "Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization and Mapping", ICRA, 2020. [[Paper](https://arxiv.org/abs/1910.02490)] [[Code](https://github.com/MIT-SPARK/Kimera-Semantics?tab=readme-ov-file)]

Kimera is an open-source library for real-time semantic visual SLAM. It allows mesh construction and semantic labelling in 3D. 


* **ConceptFusion**: "ConceptFusion: Open-set Multimodal 3D Mapping", Robotics: Science and Systems (RSS), 2023. [[Paper](https://arxiv.org/abs/2302.07241)] [[Code](https://github.com/concept-fusion/concept-fusion)] [[Website](https://concept-fusion.github.io/)]

Jatavallabhula et al. `\cite{jatavallabhula2023conceptfusion}` proposed ConceptFusion - a pixel-alighed open-set 3D mapping using foundation. The three objectives of ConceptFusion are (i) open-set (classifying data to various concepts and levels of details) and (ii) multimodal (language, image, audio, 3D geometry) using LLM and are zero-shot queryable by text, image, audio and quick queries. The system was tested on simulated and real-world datasets, tabletop manipulation, autonomous driving platform.

**Methdology**:
- Given sequence of observations $\mathcal{I}$ an open-set multimodal 3D map $\mathcal{M}$ is built. A foundation model $\mathcal{F}_{mode}$ is used as modality-specific encoder to convert query concept of specific mode (text, audio, image...) to query vector $q_{mode}$.

- Fusing pixel-aligned foundation features to 3D
Map representation: 3D map $\mathcal{M}$ is an unordered set of points, each with a vertex position, normal vector, confidence count, color vector, concept vector.

...


---
## Simulation Platforms

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

