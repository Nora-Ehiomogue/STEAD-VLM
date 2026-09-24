"""Constants shared by every stage (values as stated in the thesis, Chapters 3-4)."""

CLIP_LEN = 16            # frames per clip (STEAD-Fast input length)
SAMPLE_FPS = 5           # effective sampling rate: every 6th frame of 30 FPS footage
TARGET = 320             # spatial size fed to the X3D backbone
THETA = 0.65             # Stage-1 escalation threshold
YOLO_CONF = 0.50         # Stage-0 detection confidence
STAGE0_MIN_RELEVANT = 4  # a clip passes Stage 0 if >= this many of its 16 frames pass
SEED = 42
