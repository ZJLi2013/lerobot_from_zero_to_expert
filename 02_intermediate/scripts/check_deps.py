import sys
for mod in ["pyarrow", "cv2", "pandas", "PIL"]:
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "ok")
        print(f"{mod}: {v}")
    except ImportError:
        print(f"{mod}: NOT FOUND")
