USER_FACE_WIDTH = 140
DEFAULT_WEBCAM = 0

# Confidence Thresholds
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8 

# Eye marks
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]

# Lip marks
UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# Head indices
head_indices_pos = [1, 33, 61, 199, 263, 291]

# Eye indices
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

# Movement Threshold
movement_threshold = 0.004

# loop detection buffer
LOOP_DETECTION_FRAME_BUFFER = 500

# Audio Processing Settings
AUDIO_DEVICE_INDEX = -1  
AUDIO_FRAME_LENGTH = 512
AUDIO_VOLUME_THRESHOLD = 100  

# Captcha stuff
CAPTCHA_TIME_LIMIT = 15

# Plotting Stuff maybe not required?
MAX_PLOT_POINTS = 100
PLOT_TIME_WINDOW_SECONDS = 180 