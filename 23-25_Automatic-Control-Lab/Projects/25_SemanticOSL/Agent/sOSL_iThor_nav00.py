import time
import cv2
import ai2thor.controller

# Initialize the controller and start the simulation
controller = ai2thor.controller.Controller()

# Create an OpenCV window to display the simulation cv2imgs
cv2.namedWindow("AI2-THOR", cv2.WINDOW_NORMAL)

def nSteps(n: int = 0,
           action: str = 'MoveAhead',
           delay: float = 0.5):
    valid_actions = {"MoveAhead", "MoveBack", "MoveRight", "MoveLeft"}
    if action not in valid_actions:
        raise ValueError(f"Invalid action '{action}'. Valid actions are: {valid_actions}")

    for step in range(n):
        event = controller.step(action=action)
        print(f"Step {step+1}: {event.metadata['agent']['position']}")
        cv2.imshow("AI2-THOR", event.cv2img)
        cv2.waitKey(int(delay * 1000))

def main():
    # Small initial pause
    controller.step("PausePhysicsAutoSim")
    controller.step(
        action="AdvancePhysicsStep",
        timeStep=0.01
    )
    controller.step("UnpausePhysicsAutoSim")
    
    print('Starting simulation')

    # First rotation and display its cv2img
    event = controller.step(action='RotateRight',
                            degrees=10)
    print("Rotated Right:", event.metadata['agent']['rotation'])
    cv2.imshow("AI2-THOR", event.cv2img)
    cv2.waitKey(100)
    time.sleep(0.05)

    nSteps(4, 'MoveAhead', 0.5)

    # Rotate right again and display
    event = controller.step(action='RotateRight')
    print("Rotated Right:", event.metadata['agent']['rotation'])
    cv2.imshow("AI2-THOR", event.cv2img)
    cv2.waitKey(100)
    time.sleep(0.05)

    # Move ahead 12 steps while displaying cv2imgs
    nSteps(12, 'MoveAhead', 0.5)

    # Rotate right once more and display
    event = controller.step(action='RotateRight')
    print("Rotated Right:", event.metadata['agent']['rotation'])
    cv2.imshow("AI2-THOR", event.cv2img)
    cv2.waitKey(100)

    # Move ahead 4 steps while displaying cv2imgs
    nSteps(4, 'MoveAhead', 0.5)

    # Final rotation and display
    event = controller.step(action='RotateRight')
    print("Rotated Right:", event.metadata['agent']['rotation'])
    cv2.imshow("AI2-THOR", event.cv2img)
    cv2.waitKey(100)
    time.sleep(0.05)

    # Stop the controller and close the display window
    controller.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
