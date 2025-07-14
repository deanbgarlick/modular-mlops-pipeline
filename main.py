"""Main dispatcher for text classification experiments and pipeline execution.

This file serves as a unified entry point that can run either single experiments
or experiment suites. For more focused execution, use:
- main_single_run.py for single experiments
- main_experiment_run.py for experiment suites
"""

import sys
import argparse
from main_single_run import run_single_experiment, setup_persistence_environment as setup_single_persistence, demonstrate_persistence_features as demo_single_features
from main_experiment_run import run_experiment_suite, setup_persistence_environment as setup_exp_persistence, demonstrate_persistence_features as demo_exp_features


def main():
    """Main dispatcher for ML pipeline execution."""
    
    parser = argparse.ArgumentParser(description="Text Classification ML Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["single", "suite"], 
        default="single",
        help="Run mode: 'single' for single experiment, 'suite' for experiment suite"
    )
    parser.add_argument(
        "--show-demo", 
        action="store_true", 
        help="Show persistence features demonstration"
    )
    
    args = parser.parse_args()
    
    print("🚀 Modular ML Pipeline with Cloud Persistence")
    print("=" * 50)
    
    if args.mode == "single":
        print("📄 Mode: Single Experiment")
        print("   For dedicated single experiment execution: python main_single_run.py")
        
        # Setup persistence configuration for single run
        persistence_config = setup_single_persistence()
        
        # Show demo if requested
        if args.show_demo:
            demo_single_features()
        
        # Run single experiment
        results = run_single_experiment(persistence_config)
        
        print("\n" + "="*50)
        print("✅ Single experiment execution complete!")
        print("   Use 'python main.py --mode suite' for experiment suites")
        
    elif args.mode == "suite":
        print("📊 Mode: Experiment Suite")
        print("   For dedicated experiment suite execution: python main_experiment_run.py")
        
        # Setup persistence configuration for experiment suite
        persistence_config = setup_exp_persistence()
        
        # Show demo if requested
        if args.show_demo:
            demo_exp_features()
        
        # Run experiment suite
        results = run_experiment_suite(persistence_config)
        
        print("\n" + "="*50)
        print("✅ Experiment suite execution complete!")
        print("   Use 'python main.py --mode single' for single experiments")
    
    # Show next steps
    print("\n🔄 Next Steps:")
    print("   • Single experiments: python main_single_run.py")
    print("   • Experiment suites: python main_experiment_run.py")
    print("   • Deploy to GCP VMs: python deploy.py")
    print("   • Hyperparameter optimization: python test_hyperparameter_optimization.py")
    
    print(f"\n📝 Usage Examples:")
    print(f"   python main.py --mode single          # Run single experiment")
    print(f"   python main.py --mode suite           # Run experiment suite")
    print(f"   python main.py --mode single --show-demo  # Show persistence demo")


if __name__ == "__main__":
    main()
