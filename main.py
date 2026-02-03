"""
Main Entry Point for AI CEO Project
"""

import os
import sys

def setup_project():
    """Setup project - generate data and train models"""
    print("="*60)
    print("AI CEO PROJECT - Initial Setup")
    print("="*60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    from data.generate_sample_data import generate_all_data
    generate_all_data()
    
    # Run data pipeline
    print("\n2. Running data pipeline...")
    from preprocessing.data_pipeline import DataPipeline
    pipeline = DataPipeline()
    pipeline.load_data()
    pipeline.clean_data()
    pipeline.engineer_kpis()
    pipeline.save_master_dataset()
    
    # Train models
    print("\n3. Training ML models...")
    from models.train_models import ModelTrainer
    trainer = ModelTrainer()
    trainer.train_all()
    
    print("\n" + "="*60)
    print("✓ Setup completed successfully!")
    print("="*60)
    print("\nYou can now:")
    print("1. Run 'streamlit run dashboard/app.py' to start the dashboard")
    print("2. Run 'python orchestrator.py' to use the orchestrator API")
    print("="*60)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        setup_project()
    else:
        print("AI CEO Project")
        print("\nUsage:")
        print("  python main.py setup  - Initialize project (generate data, train models)")
        print("  streamlit run dashboard/app.py  - Start web dashboard")
        print("  python orchestrator.py  - Run orchestrator example")
