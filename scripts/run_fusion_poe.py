"""
Fusion (Product of Experts) entry point.
Later: load expert predictions, calibrate, fuse, evaluate, export figures.
"""
from digital_naturalist.paths import load_paths

def main():
    P = load_paths()
    print("Using paths:", P)

if __name__ == "__main__":
    main()
