import cv2 as cv
import numpy as np
import mediapipe as mp
import time

USER_FACE_WIDTH = 140
DEFAULT_WEBCAM = 0

MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]

UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

head_indices_pos = [1, 33, 61, 199, 263, 291]

lefteye_bottom_indices_pos = [160, 159, 158]
lefteye_top_indices_pos = [144, 145, 153]
lefteye_iris_center_indices_pos = [468]
lefteye_leftcorner_indices_pos = [158, 153]
lefteye_rightcorner_indices_pos = [160, 144]

righteye_bottom_indices_pos = [380, 374, 373]
righteye_top_indices_pos = [385, 386, 387]
righteye_iris_center_indices_pos = [473]
righteye_leftcorner_indices_pos = [387, 373]
righteye_rightcorner_indices_pos = [380, 385]

# face_looks_tot = []
# eye_looks_tot = []
# counter = 0

# prev_lip_distance = 0
# movement_threshold = 0.004
# movement_detected = False
# strike_count = 0
# last_detection_time = 0
# display_duration = 3  # duration in seconds to show detection
# cooldown_duration = 5  # cooldown duration in seconds

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)


# cap = cv.VideoCapture(r'D:\College\final_yr_prog\face_liveness_detection\loop_detect\videos\orig_vid_3.mp4')
def normalize(x, c, b1, b2):
    if x <= b1:
        return -1.0  # Clamp to 1 if x <= b1
    elif b1 < x < c:
        # Linearly map from [b1, c] -> [1, 0]
        return -(c - x) / (c - b1)
    elif c <= x < b2:
        # Linearly map from [c, b2] -> [0, 1]
        return (x - c) / (b2 - c)
    else:  # x >= b2
        return 1.0  # Clamp to 1 if x >= b2


def calculate_lip_distance(landmarks):
    upper_lip = landmarks[UPPER_LIP[5]]  # center point of upper lip
    lower_lip = landmarks[LOWER_LIP[5]]  # center point of lower lip
    return np.sqrt((upper_lip.y - lower_lip.y) ** 2)


def face_tracking(frame):
    # counter = 0

    # prev_lip_distance = 0
    # movement_threshold = 0.004
    # movement_detected = False
    # strike_count = 0
    # last_detection_time = 0
    # display_duration = 3  # duration in seconds to show detection
    # cooldown_duration = 5

    current_time = time.time()
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = mp_face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        num_faces = len(results.multi_face_landmarks) #change

        if num_faces>=2:
            print(f" MALPRACTICE IDENTIFIED: Detected {num_faces} faces!")
            return ("MULTIPLE_FACES", None)
        
        elif num_faces==1:
            
            # face_landmarks = results.multi_face_landmarks[0].landmark #change

            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )
            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )

            focal_length = 1 * img_w

            # Head direction
            head_pose_points_3D = np.multiply(
                mesh_points_3D[head_indices_pos], [img_w, img_h, 1]
            )
            head_pose_points_2D = mesh_points[head_indices_pos]

            nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
            nose_2D_point = head_pose_points_2D[0]

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # pnp_points_3D = mesh_points_3D[head_indices_pos].astype(np.float64)
            # pnp_points_2D = mesh_points[head_indices_pos].astype(np.float64)
            pnp_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
            pnp_points_3D = head_pose_points_3D.astype(np.float64)
            pnp_points_2D = pnp_points_2D.astype(np.float64)

            try:
                success, rot_vec, trans_vec = cv.solvePnP(
                    pnp_points_3D, pnp_points_2D, cam_matrix, dist_matrix
                )

                rotation_matrix, jac = cv.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

                angle_x = angles[0] * 360
                angle_y = angles[1] * 360
                z = angles[2] * 360 

                threshold_angle = 10

                if angle_y < -threshold_angle:
                    face_looks = "Right"
                elif angle_y > threshold_angle:
                    face_looks = "Left"
                elif angle_x < -threshold_angle:
                    face_looks = "Down"
                elif angle_x > threshold_angle:
                    face_looks = "Up"
                else:
                    face_looks = "Forward"

            except Exception as e:
                    print(f"Error during solvePnP or angle calculation: {e}")
                    face_looks = "Error"
                    angle_x, angle_y = 0, 0

            # Eye Direction
            try:
                lefteye_top_points = mesh_points[lefteye_top_indices_pos]
                lefteye_bot_points = mesh_points[lefteye_bottom_indices_pos]
                lefteye_center_pos = np.mean(np.concatenate((lefteye_bot_points, lefteye_top_points), axis=0), axis=0)
                lefteye_iris_center_pos = mesh_points[lefteye_iris_center_indices_pos]
                lefteye_rightcorner_pos = np.mean(mesh_points[lefteye_rightcorner_indices_pos], axis=0)
                lefteye_leftcorner_pos = np.mean(mesh_points[lefteye_leftcorner_indices_pos], axis=0)

                righteye_top_points = mesh_points[righteye_top_indices_pos]
                righteye_bot_points = mesh_points[righteye_bottom_indices_pos]
                righteye_center_pos = np.mean(np.concatenate((righteye_bot_points, righteye_top_points), axis=0), axis=0)
                righteye_iris_center_pos = mesh_points[righteye_iris_center_indices_pos]
                righteye_rightcorner_pos = np.mean(mesh_points[righteye_rightcorner_indices_pos], axis=0)
                righteye_leftcorner_pos = np.mean(mesh_points[righteye_leftcorner_indices_pos], axis=0)

                if (lefteye_leftcorner_pos[0] - lefteye_rightcorner_pos[0]) != 0 and \
                        (righteye_leftcorner_pos[0] - righteye_rightcorner_pos[0]) != 0:

                    normalized_lefteye = normalize(lefteye_iris_center_pos[0][0], lefteye_center_pos[0], lefteye_rightcorner_pos[0],
                                                lefteye_leftcorner_pos[0])
                    normalized_righteye = normalize(righteye_iris_center_pos[0][0], righteye_center_pos[0],
                                                    righteye_rightcorner_pos[0], righteye_leftcorner_pos[0])

                    if normalized_lefteye > 0 and normalized_lefteye > 0:
                        eye_looks = 'center'
                        if (normalized_lefteye + normalized_righteye) / 2 > 0.5:
                            eye_looks = 'left'
                    elif normalized_lefteye < 0 and normalized_lefteye < 0:
                        eye_looks = 'center'
                        if (normalized_lefteye + normalized_righteye) / 2 < -0.5:
                            eye_looks = 'right'
                    else:
                        eye_looks = 'center'
                else:
                    eye_looks = 'Center'

            except Exception as e:
                    print(f"Error calculating eye direction: {e}")
                    eye_looks = "Error"

        # calc lip distance

        # for face_landmarks in results.multi_face_landmarks:
        #     landmarks = face_landmarks.landmark
        # lip_distance = calculate_lip_distance(landmarks)

        # detecting lip movement
        # if prev_lip_distance > 0:
        #     movement = lip_distance - prev_lip_distance
        #     if movement > movement_threshold and (current_time - last_detection_time) > cooldown_duration:
        #         movement_detected = True
        #         strike_count += 1
        #         last_detection_time = current_time

        # prev_lip_distance = lip_distance

        # return [mesh_points, landmarks, face_looks, eye_looks, movement_detected, strike_count, nose_2D_point, angle_x,
        #         angle_y, lip_distance]

            try:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                lip_distance = calculate_lip_distance(landmarks)

            except Exception as e:
                 print(f"Error calculating lip distance: {e}")
                 lip_distance = 0 # Default on error

            detailed_data = [mesh_points, face_landmarks, face_looks, eye_looks,
                             nose_2D_point, angle_x, angle_y, lip_distance]

            
            return ("OK", detailed_data)
    else:
        return ("NO_FACE", None)