from detection import process_live_video, X, O

MODEL_PATH = r"..\runs\detect\train3\weights\best.pt"

process_live_video(1, MODEL_PATH, user=X)