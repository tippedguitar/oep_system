import cv2
import threading
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from cvzone.HandTrackingModule import HandDetector
from loop_detect.detect import VideoLoopFinder
from pvrecorder import PvRecorder
from gaze_direction.face_Direction import face_tracking
from matplotlib.animation import FuncAnimation


from constants import *

class OEP():
    def __init__(self):
        
        self.loop_finder = VideoLoopFinder()
        self.input = True

        # loop stuff
        self.current_frames = []
        self.FRAME = LOOP_DETECTION_FRAME_BUFFER
        self.check_for_loop = True
        self.is_loop = False
        self.loop_score = 0
        self.loop_check_count = 0

        # captcha stuff
        self.frame = None
        self.captcha_running = False
        self.detector = HandDetector(maxHands=3, detectionCon=MIN_DETECTION_CONFIDENCE)
        self.timelimit = CAPTCHA_TIME_LIMIT
        self.lock = threading.Lock()
        self.show_text = ''
        self.remainingTime = 0
        self.required_gesture = 0
        self.two_hands_warning_count = 0

        # face/eye/lip stuff
        self.tracking = True
        self.mesh_points = None
        self.landmarks = None
        self.nose_2D_point = None
        self.face_looks = ''
        self.eye_looks = ''
        self.movement_detected = '' 
        self.strike_count = ''      
        self._SHOW_GAZE = False
        self._SHOW_EYE = False
        self._SHOW_LIPS = False
        self.angle_x = 0
        self.angle_y = 0
        self.movement_count = 0
        self.current_lip = 0
        self.prev_lip = 0

        # audio stuff
        self.device_index = AUDIO_DEVICE_INDEX
        self.frame_length = AUDIO_FRAME_LENGTH
        self.volume_threshold = AUDIO_VOLUME_THRESHOLD
        try:
            self.audio_input = PvRecorder(device_index=self.device_index, frame_length=self.frame_length)
            self.audio_available = True
        except Exception as e:
            print(f"Error initializing audio recorder: {e}")
            print("Audio processing will be disabled.")
            self.audio_input = None
            self.audio_available = False

        self.audio_alert = ""
        self.check_audio = self.audio_available
        self.audio_frame = None
        self.current_audio = 0

        # plotting
        self.time = 0
        self.audio_xs = []
        self.audio_ys = []
        self.threshold_ys = []
        self.MAX_POINTS = MAX_PLOT_POINTS

        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Real-Time Audio Volume Plot")
        self.ax.set_xlabel("Time (Frame Count)")
        self.ax.set_ylabel("Volume (RMS)")
        self.line, = self.ax.plot([], [], lw=2, label='Volume')
        self.thresh_line, = self.ax.plot([], [], lw=1, color='red', linestyle='--', label='Threshold')
        self.ax.legend()

    # Loop Detection
    def loop_thread(self):
        print("Loop detection thread started")
        while self.input:
            if self.check_for_loop:
                process_frames = False
                frames_to_process = []
                with self.lock:
                    if len(self.current_frames) >= self.FRAME and not self.captcha_running:
                        process_frames = True
                        frames_to_process = self.current_frames[:]
                        self.current_frames = []

                if process_frames:
                    try:
                        self.loop_check_count += 1
                        duplicate_frames = self.loop_finder.find_duplicates(frames_to_process)
                        isLooped, loop_frame_count = self.loop_finder.get_valid_duplicates(duplicate_frames)
                        self.loop_score += loop_frame_count

                        if isLooped and not self.captcha_running:
                             with self.lock:
                                self.is_loop = True
                             self.start_captcha_thread()
                        else:
                            avg_score = self.loop_score / self.loop_check_count if self.loop_check_count > 0 else 0
                            print(f"Loop Analysis -> Score: {self.loop_score}, Checks: {self.loop_check_count}, Avg: {avg_score:.2f}")
                        print(f"Loop Detected in batch: {isLooped}")
                    except Exception as e:
                         print(f"Error in loop detection analysis: {e}")

                time.sleep(0.1)
            else:
                 time.sleep(0.5)
        print("Loop detection thread finished.")


    #Captcha 
    def generate_required_gesture(self):
        return random.randint(1, 5)

    def captcha_thread(self):
        initialTime = time.time()
        requiredGesture = self.generate_required_gesture()
        print(f"Captcha started. Required gesture: {requiredGesture}")
        base_prompt = f'Show {requiredGesture} fingers (Use ONE hand)'

        with self.lock:
            self.show_text = base_prompt
            self.remainingTime = self.timelimit
            self.check_for_loop = False
            self.required_gesture = requiredGesture
            self.captcha_running = True
            self._SHOW_EYE = False
            self._SHOW_GAZE = False
            self._SHOW_LIPS = False

        accepted = False
        malpractice_detected = False
        while True:
            timer = time.time() - initialTime
            current_remaining_time = self.timelimit - timer

            if current_remaining_time <= 0:
                print("Captcha Rejected! Time Over.")
                with self.lock:
                    self.show_text = "Rejected! Time Over"
                time.sleep(2)
                break

            local_frame_copy = None
            with self.lock:
                self.remainingTime = current_remaining_time
                if self.frame is not None:
                    local_frame_copy = self.frame.copy()

            if local_frame_copy is None:
                time.sleep(0.1)
                continue

            try:
                hands, _ = self.detector.findHands(local_frame_copy, draw=False)
                num_hands = len(hands) if hands else 0

                if num_hands >= 3:
                    
                    malpractice_msg = "Malpractice Identified! (>=3 Hands)"
                    print(f"MALPRACTICE: {num_hands} hands detected!")
                    with self.lock:
                        self.show_text = malpractice_msg
                        self.input = False
                    malpractice_detected = True
                    time.sleep(2)
                    break 

                elif num_hands == 2:
                    warning_msg = "Two hands seen. Use only ONE hand."
                    print("CAPTCHA WARNING: Exactly 2 hands detected.")
                    with self.lock:
                        if self.show_text != warning_msg:
                             self.two_hands_warning_count += 1
                             print(f"Two hands warning count triggered: {self.two_hands_warning_count}")
                        self.show_text = warning_msg

                elif num_hands == 1:
                    hand = hands[0]
                    fingers = self.detector.fingersUp(hand)
                    fingersCount = fingers.count(1)
                    finger_count_text = f'Fingers: {fingersCount}'

                    just_corrected_from_two = False
                    with self.lock:
                         if self.show_text == "Two hands seen. Use only ONE hand.":
                              just_corrected_from_two = True

                    with self.lock:
                        if just_corrected_from_two or self.show_text == base_prompt:
                            self.show_text = finger_count_text
                        elif self.show_text not in ["Accepted!", "Rejected! Time Over",
                                                   "Malpractice Identified! (>=3 Hands)",
                                                   "Two hands seen. Use only ONE hand."]:
                             self.show_text = finger_count_text


                    if fingersCount == requiredGesture:
                        print("Captcha Accepted!")
                        with self.lock:
                            self.show_text = "Accepted!"
                        accepted = True
                        time.sleep(1)
                        break

                else: 
                    with self.lock:
                         if self.show_text not in ["Accepted!", "Rejected! Time Over",
                                                  "Malpractice Identified! (>=3 Hands)",
                                                  "Two hands seen. Use only ONE hand."]:
                              self.show_text = base_prompt
                         elif self.show_text == "Two hands seen. Use only ONE hand.":
                             self.show_text = base_prompt

            except Exception as e:
                print(f"Error processing hand detection in captcha: {e}")

            time.sleep(0.1)

        with self.lock:
            if not malpractice_detected:
                 self.show_text = ''
            if self.input:
                 self.check_for_loop = True
            self.captcha_running = False
            if accepted:
                 self.is_loop = False
            print("Captcha finished.")

    def start_captcha_thread(self):
        if not self.captcha_running:
            print("Starting captcha thread....")
            captcha_proc_thread = threading.Thread(target=self.captcha_thread, daemon=True)
            captcha_proc_thread.start()
        else:
            print("Captcha thread already running.")

    # Audio 
    def calculate_rms(self, frame_data):
        rms = np.sqrt(np.mean(np.square(frame_data.astype(np.float64))))
        return rms

    def audio_thread(self):
        if not self.audio_available:
            print("Audio thread not started (device unavailable).")
            return

        print("Audio processing thread started.....")
        self.audio_input.start()
        temp_count = 0 
        flag = 1       
        avg_vol = 0    
        avg_counter = 0

        try:
            while self.check_audio and self.input: 
                try:
                    pcm = self.audio_input.read()
                    audio_frame_data = np.array(pcm, dtype=np.int16)
                except Exception as e:
                    print(f"Error reading audio frame: {e}")
                    time.sleep(0.1) 
                    continue

                try:
                    volume = self.calculate_rms(audio_frame_data)
                    avg_vol += volume
                    avg_counter += 1

                    
                    with self.lock:
                        self.current_audio = volume 

                        self.time += 1
                        self.audio_xs.append(self.time)
                        self.audio_ys.append(volume)
                        self.threshold_ys.append(self.volume_threshold)

                        if len(self.audio_xs) > self.MAX_POINTS:
                             self.audio_xs.pop(0)
                             self.audio_ys.pop(0)
                             self.threshold_ys.pop(0)

                        if volume > self.volume_threshold:
                            if self.audio_alert != "Noise Detected":
                                print(f"ALERT: Noise detected (Volume: {volume:.2f})")
                            self.audio_alert = "Noise Detected"
                            temp_count = temp_count + 1 if flag == 1 else 0
                            flag = 1
                        else:
                            if self.audio_alert == "Noise Detected":
                                print("ALERT: Noise level normal.")
                            self.audio_alert = "No noise detected"
                            temp_count = temp_count + 1 if flag == 0 else 0
                            flag = 0

                except Exception as e:
                    print(f"Error processing audio frame: {e}")


        finally:
            if self.audio_available and self.audio_input:
                self.audio_input.stop()
                self.audio_input.delete()
            print("Audio processing thread finished, resources released.")

    # Face Tracking 

    # def tracking_thread(self):
    #     print("Tracking thread started.....")
    #     while self.tracking and self.input: 
    #         local_frame_copy = None
    #         with self.lock:
    #             if self.frame is not None:
    #                 local_frame_copy = self.frame.copy() 

        #     if local_frame_copy is not None:
        #         try:
        #             
        #             tracking_results = face_tracking(local_frame_copy)
                    

        #             if tracking_results is not None:
        #                 
        #                 (mesh_pts, landmarks_data, face_dir, eye_dir, move_detect,
        #                  strikes, nose_pt, angle_x_val, angle_y_val, lip_dist) = tracking_results

        #                 
        #                 with self.lock:
        #                     self.prev_lip = self.current_lip
        #                     self.mesh_points = mesh_pts
        #                     self.landmarks = landmarks_data
        #                     self.face_looks = face_dir
        #                     self.eye_looks = eye_dir
        #                     self.movement_detected = move_detect 
        #                     self.strike_count = strikes        
        #                     self.nose_2D_point = nose_pt
        #                     self.angle_x = angle_x_val
        #                     self.angle_y = angle_y_val
        #                     self.current_lip = lip_dist

        #                     if self.current_lip - self.prev_lip > movement_threshold:
        #                         self.movement_count += 1
        #             else:
        #                  with self.lock:
        #                       self.face_looks = 'No Face Detected'
        #                       self.eye_looks = ''
        #                       self.mesh_points = None
        #                       self.landmarks = None
        #         except Exception as e:
        #             print(f"Error in face tracking function call: {e}")

        #     time.sleep(0.05)
        # print("Tracking thread finished.")
        
    def tracking_thread(self):
        print("Tracking thread started.....")
        while self.tracking and self.input: 
            local_frame_copy = None

            with self.lock:
                 if not self.input:
                      break
                 if self.frame is not None:
                      local_frame_copy = self.frame.copy()

            if local_frame_copy is not None:
                try:
                    tracking_status, tracking_data = face_tracking(local_frame_copy)

                    if tracking_status == "MULTIPLE_FACES":
                         # MALPRACTICE: MULTIPLE FACES DETECTED
                         print("MALPRACTICE (Tracking): Multiple faces detected! Stopping system.")
                         with self.lock:
                              self.input = False
                              self.face_looks = "MULTIPLE FACES!"
                              self.eye_looks = ''
                              self.mesh_points = None
                              self.landmarks = None
                         break

                    elif tracking_status == "OK":
                         with self.lock:
                              (self.mesh_points, self.landmarks, self.face_looks,
                               self.eye_looks, self.nose_2D_point, self.angle_x,
                               self.angle_y, current_lip_val) = tracking_data

                              if self.prev_lip is None:
                                   self.prev_lip = current_lip_val

                              self.prev_lip = self.current_lip 
                              self.current_lip = current_lip_val 

                              if self.prev_lip is not None and self.prev_lip > 0:
                                    if self.current_lip - self.prev_lip > movement_threshold:
                                         self.movement_count += 1 
                                         # print(f"Lip movement detected! Count: {self.movement_count}")

                    elif tracking_status == "NO_FACE":
                         with self.lock:
                              if self.face_looks != 'No Face Detected':
                                   print("Tracking: No face detected.")
                              self.face_looks = 'No Face Detected'
                              self.eye_looks = ''
                              self.mesh_points = None
                              self.landmarks = None

                except Exception as e:
                    print(f"Error in face tracking thread processing: {e}")

            time.sleep(0.05)

        print("Tracking thread finished.")

    # Main Loop Calls 
    def extract_frames(self, video_source):
        video = cv2.VideoCapture(video_source)
        if not video.isOpened():
            print(f"Error: Could not open video source {video_source}")
            self.input = False
            return

        cv2.namedWindow("Proctoring System", cv2.WINDOW_NORMAL)

        threading.Thread(target=self.loop_thread, daemon=True).start()
        threading.Thread(target=self.audio_thread, daemon=True).start()
        threading.Thread(target=self.tracking_thread, daemon=True).start()

        print("Starting main video processing loop...")
        while self.input:
            try:
                success, local_frame = video.read()
                if not success:
                    print("End of video source or cannot read frame.")
                    break

                img_h, img_w = local_frame.shape[:2]

                with self.lock:
                    self.frame = local_frame.copy() 

                    if self.check_for_loop and not self.captcha_running:
                         if len(self.current_frames) < self.FRAME:
                             self.current_frames.append(self.frame)

                display_frame = local_frame.copy()

                # --- Get State Variables under Lock ---
                with self.lock:
                    show_text_val = self.show_text
                    required_gesture_val = self.required_gesture
                    remaining_time_val = self.remainingTime
                    is_checking_loop = self.check_for_loop
                    is_captcha_running = self.captcha_running
                    loop_check_count_val = self.loop_check_count
                    is_loop_detected_flag = self.is_loop 
                    audio_alert_val = self.audio_alert
                    volume_thresh_val = self.volume_threshold
                    show_gaze = self._SHOW_GAZE
                    show_eye = self._SHOW_EYE
                    show_lips = self._SHOW_LIPS
                    mesh_points_val = self.mesh_points.copy() if self.mesh_points is not None else None
                    landmarks_val = self.landmarks 
                    nose_2d_val = self.nose_2D_point
                    angle_x_val = self.angle_x
                    angle_y_val = self.angle_y
                    eye_looks_val = self.eye_looks
                    face_looks_val = self.face_looks
                    movement_count_val = self.movement_count


                # Display Frame and text to be shown
                # Captcha Text
                if show_text_val:
                    cv2.putText(display_frame, f"Action: {show_text_val}", (img_w // 2 - 150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if "remaining time" not in show_text_val.lower() and required_gesture_val != 0:
                         cv2.putText(display_frame, f"Time Left: {max(0, round(remaining_time_val, 1))}s", (img_w // 2 - 150, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Status Text
                status_y = 30
                if is_captcha_running:
                    status_text = "CAPTCHA RUNNING"
                    status_color = (0, 165, 255)
                    if is_loop_detected_flag:
                         status_text += " (Loop Detected)"
                    else:
                         status_text += " (Manual Trigger)"
                elif is_checking_loop:
                    status_text = f"Monitoring... (Loop Check {loop_check_count_val})"
                    status_color = (0, 255, 0)
                else:
                    status_text = "Status Unknown"
                    status_color = (255, 0, 0)
                cv2.putText(display_frame, status_text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                status_y += 25

                # Audio
                if self.audio_available:
                    audio_color = (0, 0, 255) if "Noise Detected" in audio_alert_val else (0, 255, 0)
                    cv2.putText(display_frame, f"Audio: {audio_alert_val} (Thresh: {volume_thresh_val})", (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 2)
                    status_y += 25


                # Tracking Info
                if not is_captcha_running:
                    track_y = img_h - 100 
                    if show_gaze or face_looks_val:
                        cv2.putText(display_frame, f"Face Direction: {face_looks_val}", (10, track_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        track_y += 25
                    if show_eye or eye_looks_val:
                         cv2.putText(display_frame, f"Eye Direction: {eye_looks_val}", (10, track_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                         track_y += 25
                    if show_lips:
                         cv2.putText(display_frame, f"Lip Movement Count: {movement_count_val}", (10, track_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                         track_y += 25


                # Draw Visualizations (Eyes, Gaze, Lips) - Only if not in captcha
                if not is_captcha_running:
                    # Eye Iris/Corners 
                    if show_eye and mesh_points_val is not None:
                        try:
                            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points_val[lefteye_iris_center_indices_pos])
                            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points_val[righteye_iris_center_indices_pos])
                            center_left = np.array([l_cx, l_cy], dtype=np.int32)
                            center_right = np.array([r_cx, r_cy], dtype=np.int32)
                            cv2.circle(display_frame, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
                            cv2.circle(display_frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
                            cv2.circle(display_frame, tuple(mesh_points_val[LEFT_EYE_INNER_CORNER][0]), 2, (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(display_frame, tuple(mesh_points_val[LEFT_EYE_OUTER_CORNER][0]), 2, (0, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(display_frame, tuple(mesh_points_val[RIGHT_EYE_INNER_CORNER][0]), 2, (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(display_frame, tuple(mesh_points_val[RIGHT_EYE_OUTER_CORNER][0]), 2, (0, 255, 255), -1, cv2.LINE_AA)
                        except Exception as e:
                            # print(f"Error drawing eyes: {e}")
                            pass

                    # Gaze Direction
                    if show_gaze and nose_2d_val is not None and mesh_points_val is not None:
                        try:
                            # p1 = tuple(np.array(nose_2d_val, dtype=int))
                            # p2 = (int(nose_2d_val[0] + angle_y_val * 30), int(nose_2d_val[1] - angle_x_val * 30))

                            p1 = nose_2d_val
                            p2 = (
                                int(nose_2d_val[0] + angle_y_val * 10),
                                int(nose_2d_val[1] - angle_x_val * 10)
                            )

                            cv2.line(display_frame, p1, p2, (0, 255, 255), 2)

                            for idx in head_indices_pos:
                                 if idx < len(mesh_points_val):
                                      cv2.circle(display_frame, tuple(mesh_points_val[idx]), 1, (0, 255, 0), -1)
                        except Exception as e:
                            # print(f"Error drawing gaze: {e}")
                            pass

                    # Lip Outline
                    if show_lips and landmarks_val is not None:
                        try:
                             valid_upper_lip_indices = [i for i in UPPER_LIP if i < len(landmarks_val)]
                             valid_lower_lip_indices = [i for i in LOWER_LIP if i < len(landmarks_val)]

                             if valid_upper_lip_indices and valid_lower_lip_indices:
                                 upper_lip_pts = np.array([(int(landmarks_val[i].x * img_w), int(landmarks_val[i].y * img_h)) for i in valid_upper_lip_indices], dtype=np.int32)
                                 lower_lip_pts = np.array([(int(landmarks_val[i].x * img_w), int(landmarks_val[i].y * img_h)) for i in valid_lower_lip_indices], dtype=np.int32)
                                 cv2.polylines(display_frame, [upper_lip_pts], isClosed=True, color=(0, 255, 0), thickness=1)
                                 cv2.polylines(display_frame, [lower_lip_pts], isClosed=True, color=(0, 255, 0), thickness=1)
                        except Exception as e:
                             # print(f"Error drawing lips: {e}")
                             pass


                # Final Frame Displaay
                # cv2.namedWindow("Proctoring System", cv2.WINDOW_NORMAL)
                cv2.imshow("Proctoring System", display_frame)

                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("'q' pressed, exiting...")
                    self.input = False
                    break

                if not is_captcha_running:
                    if key == ord(' '): 
                        print("Spacebar pressed, manually starting captcha...")
                        self.start_captcha_thread()

                    elif key == ord('f'): 
                        with self.lock:
                            self._SHOW_GAZE = not self._SHOW_GAZE
                            self._SHOW_EYE = False
                            self._SHOW_LIPS = False
                            print(f"Toggled Face/Gaze: {self._SHOW_GAZE}")

                    elif key == ord('e'): 
                        with self.lock:
                            self._SHOW_EYE = not self._SHOW_EYE
                            self._SHOW_GAZE = False
                            self._SHOW_LIPS = False
                            print(f"Toggled Eye: {self._SHOW_EYE}")

                    elif key == ord('l'): 
                        with self.lock:
                            self._SHOW_LIPS = not self._SHOW_LIPS
                            self._SHOW_GAZE = False
                            self._SHOW_EYE = False
                            print(f"Toggled Lips: {self._SHOW_LIPS}")

            except Exception as e:
                print(f"Error in main loop: {e}")


        # Kind of like finally
        print("Cleaning up resources...")
        self.input = False
        self.tracking = False
        self.check_audio = False
        video.release()
        cv2.destroyAllWindows()
        print("Video released and windows closed.")