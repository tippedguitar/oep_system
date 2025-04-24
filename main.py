#! C:\Users\Srinitish\Desktop\college\7seventhsem\Finalyrproject\Project\oep_system\myenv\Scripts\python.exe

import sys
import os
import time
import cv2


from oep_system import OEP
from constants import DEFAULT_WEBCAM

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
    run_proctoring_system()
    print("Application ended")
