"""
Visual expert (ResNet-18) entry point.
Later: train multiple runs, save checkpoints, evaluate, save outputs.
"""
from digital_naturalist.paths import load_paths

def main():
    P = load_paths()
    print("IMAGE_DIR:", P["IMAGE_DIR"])
    print("MODEL_DIR:", P["MODEL_DIR"])

if __name__ == "__main__":
    main()
