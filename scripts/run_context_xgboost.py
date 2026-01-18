"""
Context expert (XGBoost) entry point.
Later: load GBIF, build features, train/evaluate, save outputs.
"""
from digital_naturalist.paths import load_paths

def main():
    P = load_paths()
    print("GBIF_DIR:", P["GBIF_DIR"])
    print("MODEL_DIR:", P["MODEL_DIR"])

if __name__ == "__main__":
    main()
