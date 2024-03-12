## March 11, 2024

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
