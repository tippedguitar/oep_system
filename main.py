#! C:\Users\Srinitish\Desktop\college\7seventhsem\Finalyrproject\Project\oep_system\myenv\Scripts\python.exe

import sys
import os
import time
import cv2


from oep_system import OEP
from constants import DEFAULT_WEBCAM

# Path to the UI images
UI_FOLDER = "ui"
SPLASH_SCREEN_IMAGE = os.path.join(UI_FOLDER, "splash_screen.png")
START_BUTTON_IMAGE = os.path.join(UI_FOLDER, "start_button.png")

def show_splash_screen():
    # Load the splash screen and button images
    splash_image = cv2.imread(SPLASH_SCREEN_IMAGE)
    button_image = cv2.imread(START_BUTTON_IMAGE, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

    if splash_image is None:
        print(f"Error: Unable to load splash screen image from {SPLASH_SCREEN_IMAGE}")
        return
    if button_image is None:
        print(f"Error: Unable to load button image from {START_BUTTON_IMAGE}")
        return

    # Resize the splash screen image to fit the window (1170x760)
    splash_image = cv2.resize(splash_image, (1170, 760))

    # Resize the button image to 50% of its original size
    button_scale = 0.5  # Scale factor for the button
    button_image = cv2.resize(button_image, (0, 0), fx=button_scale, fy=button_scale)

    # Resize the window to 1170x760
    window_name = "Splash Screen"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1170, 760)

    # Overlay the button on the splash screen (bottom-left corner)
    button_height, button_width = button_image.shape[:2]
    button_x = 50 + 23  # 50 pixels from the left + 38 pixels (1 cm to the right)
    button_y = splash_image.shape[0] - button_height - 50  # 50 pixels from the bottom

    # Extract the alpha channel from the button image
    button_alpha = button_image[:, :, 3] / 255.0
    button_rgb = button_image[:, :, :3]

    # Blend the button into the splash screen
    for c in range(0, 3):
        splash_image[button_y:button_y + button_height, button_x:button_x + button_width, c] = (
            splash_image[button_y:button_y + button_height, button_x:button_x + button_width, c] * (1 - button_alpha) +
            button_rgb[:, :, c] * button_alpha
        )

    # Display the splash screen with the button
    cv2.imshow(window_name, splash_image)

    # Mouse callback function to detect button clicks
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the click is within the button area
            if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
                cv2.destroyAllWindows()  # Close the splash screen
                run_proctoring_system()  # Start the proctoring system

    # Set the mouse callback function
    cv2.setMouseCallback(window_name, on_mouse_click)

    # Wait for a click or until the user closes the window
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def run_proctoring_system():
    print("Starting Proctoring System")
    createdObject = OEP()

    video_source = DEFAULT_WEBCAM
    # video_source = r''

    # print(f"Using video source: {video_source}")

    try:
        createdObject.extract_frames(video_source=video_source)

    except KeyboardInterrupt:
        print("\nShutting down")
        createdObject.input = False
        time.sleep(1)

    except Exception as e:
        print("error occurred in main execution")
        print(f"Error: {e}")
        # import traceback
        # traceback.print_exc()
        # createdObject.input = False 

    finally:
        print("Main function finished")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    show_splash_screen()  # Show splash screen with button
    print("Application ended")
