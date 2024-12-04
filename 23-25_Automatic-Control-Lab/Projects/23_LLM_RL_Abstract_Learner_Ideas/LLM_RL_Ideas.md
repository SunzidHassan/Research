## March 11, 2024


I prefer the latest Intel RealSense cameras instead of R200. Then, we will need some successful examples of using the latest model (e.g., D400) on ROS  1 (or turtlebot3).


From the information that I collected, it seems like the depth camera is doing the same thing as the LSD, to measure the object distance from the ego robot. So, some new questions raise:

 

Find some successful examples of using the latest Intel RealSense cameras (e.g., D400) on ROS 1. You can use this link as the starting point: https://wiki.ros.org/IntelROSProject
Find some existing applications (or tutorials) that use the depth camera to (i) detect an object from the image (ii) show the object distance to the ego robot (iii) calculate the object position in the global frame.
Once we have a camera, how to install it on the Turtlebot3 (through usb cable? Raspberry pi camera cable? Or anything else?)


A possible publication opportunity is to write survey paper, centering on LLM in robotic tasks (object manipulation, autonomous driving, multi-agent collaborations, etc. ) For each category, collect papers and compare them in a table. We can talk more about this idea tomorrow. You can use my paper collection as a starting point to write this review paper.



* Task 1:
    * Obtain a Multi-modal LLM that:
        * Input: Image (multiple segmentations)
        * Output: Left or Right (the direction in which the robot should go next to find the odor source)
    * Key challenges:
        * Chain of thought: how to create a prompt to indicate the LLM to reason the odor source from the input images?
        * In context learning: provide several correct examples to the LLM, and ask a new question to LLM
            * For instance, you can provide some images with the odor source on left or right side of the image, and provide correct answers (if the odor source is at left, go left; if the odor source is at right, go right).
            * Provide a description of reasoning process:
            * For instance:
            * Since I see a humidifier on the left side of the image, which means the humidifier is at the left side of my position. So I should turn left to approach the odor source location.
        * TRY THIS METHOD FIRST.

* Task 2:
    * Create the language map (a representation map of the environment that ingetrates semantic description of the surrounding objects)
    - Question 1: how to generate a language map?
        - Solution 1:
            - Integrate CLIP-seg and LDS
            - CLIP-seg provides image segmentation results based on input categories. 
        - Solution 2:
    - Question 2: How to get the relevance score?
        - Solution 1:
        - Solution 2:


## March 14, 2024


### LLM Review Paper
(8-10 pages, 12 pages at most) aim to publish on a journal similar to Sensor. 

LLM Review paper: Embodied AI Agent in Robotics and Autonomous Systems

10-12 pages, comprehensive citation (at least 200 citation)

- Background: Start with development of LLMs
    - Start with transformer architecture (attention is all you need).
    - LLMs like BERT, GPT, BLIP, T5.
    - Multimodal LLMs - GPT4, CLIP, Flamingo, VisualBERT, BLIP-2, Delle2, Palm-e.

- LLM in robotic applicaiton:
    - Object grounding: 
    - Robotic transformer: RT-1, RT-2, decisionformer, Galo.
    - Robot navigation: https://github.com/lingxiaoW/PaperNotes/blob/main/Paper%20Reading/Robotics/AI%20Robots%20Integration/AIRobotsIntegration.md
    - Dataset, simulation platform, benchmarks.

- LLM in Autonomous driving
    - Dataset, simulation platform, benchmarks.

<!-- - Dataset, simulation platform, benchmarks.
    - Dataset: seperate table to summarise different dataset in terms of robotic tasks (with mention of used hardware and inference time or unavailable). -->

- Challenges, future direction
    - Challenges: inference time too long, most LLMs can't be deployed on local machine, 
    - Future directions: methods to mitigate the challenges
        - To make LLMs lightweight for faster inference
        - General dataset that includes diverse robotic platforms and robotic tasks (cite RT-X paper for this)
        - ...
    

- Start with defining broad (e.g., autonomous driving, robot control, object manipulation. etc.) categories of LLM based agent.


### LLM based vision and olfaction fused odor source localization

- Gazebo customized environment for turtlebot
    - Python read data from Gazebo
    - Depth camera in Gazebo 
    - [Depth camera package](https://wiki.ros.org/IntelROSProject)
    - Can Gazebo provide front-face camera images? 
- Gaden simulator to generate plume, check this [tutorial](https://colab.research.google.com/drive/1Xj7rrsmeDa_dS3Ru_UIhhzlaifGH6GS4)
- Verify the vision part 
    - CLIP-seg (try with real images from the previous tests)
    - LLM to assign rewards to different objects

