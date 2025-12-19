from src.model.main import main as run_model_training
from src.model.evaluate_test_set import evaluate_final_predictions
from src.model.plots import generate_all_plots


def main():
    print("STEP 1: Training Models and Generating Predictions")
    training_history, df_labels = run_model_training()
    
    print("\n")
    print("STEP 2: Evaluating Predictions Against Ground Truth")
    evaluate_final_predictions()
    
    print("\n")
    print("STEP 3: Generating Visualization Plots")
    generate_all_plots(
        training_history=training_history,
        predictions_path="final_predictions.csv",
        gt_df=df_labels
    )
    
    print("\n")
    print("Model Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
