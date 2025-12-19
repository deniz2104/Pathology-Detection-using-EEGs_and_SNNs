import argparse
from src.eeg_preprocessing_pipeline_entry_point import main as run_preprocessing
from src.model_pipeline_entry_point import main as run_model_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the full EEG analysis pipeline")
    parser.add_argument(
        "--skip-preprocessing", 
        action="store_true", 
        help="Skip the EEG preprocessing step (use if data is already preprocessed)"
    )
    parser.add_argument(
        "--skip-training", 
        action="store_true", 
        help="Skip the model training step"
    )
    args = parser.parse_args()
    
    if not args.skip_preprocessing:
        print("STEP 1: Running EEG Preprocessing Pipeline")
        run_preprocessing()
        print("\nPreprocessing completed!\n")
    else:
        print("Skipping preprocessing step (--skip-preprocessing flag set)")
    
    if not args.skip_training:
        print("STEP 2: Running Model Training and Inference Pipeline")
        run_model_pipeline()
        print("\nModel pipeline completed!\n")
    else:
        print("Skipping training step (--skip-training flag set)")
    
    print("Pipeline finished successfully!")

if __name__ == "__main__":
    main()
