# Full-screen OpenCV window size.
WINDOW_WIDTH_PX = 1470
WINDOW_HEIGHT_PX = 956

# Physical display geometry used to convert pixels to visual-angle degrees.
SCREEN_WIDTH_MM = 290.3
SCREEN_HEIGHT_MM = 188.7
VIEWING_DISTANCE_MM = 430.0
SCREEN_GEOMETRY_NOTE = (
    "Display 290.3 x 188.7 mm; estimated eye-to-screen distance 430 mm."
)

# Calibration: 15 points = 3 columns x 5 rows, repeated twice.
CALIBRATION_X_LEVELS = (0.12, 0.50, 0.88)
CALIBRATION_Y_LEVELS = (0.08, 0.29, 0.50, 0.71, 0.92)
CALIBRATION_ROUNDS = 2
CALIBRATION_SETTLE_SEC = 0.90
CALIBRATION_RECORD_SEC = 1.20
CALIBRATION_MIN_VALID_SAMPLES = 18

# A short neutral-head reference improves consistency during calibration.
HEAD_REFERENCE_FRAMES = 25
HEAD_STABLE_SEC = 0.30
HEAD_PITCH_TOLERANCE_DEG = 3.0
HEAD_YAW_TOLERANCE_DEG = 3.0
HEAD_ROLL_TOLERANCE_DEG = 3.0
HEAD_CENTER_TOLERANCE = 0.035
HEAD_WIDTH_TOLERANCE_FRACTION = 0.05

# Exactly one ML model is used: StandardScaler + multi-output Ridge.
RIDGE_ALPHA = 10.0

# Virtual keyboard interaction settings.
KEYBOARD_DWELL_SEC = 1.0
KEYBOARD_REARM_DISTANCE_PX = 75.0
SMOOTHING_HISTORY = 5
