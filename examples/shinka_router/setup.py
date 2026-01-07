"""
Quick setup and run script for ShinkaRouter on AIME.
"""
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check required packages."""
    required = ['pandas', 'matplotlib', 'python-dotenv', 'tqdm']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


def check_files():
    """Check required files exist."""
    required = [
        'examples/shinka_router/router.py',
        'examples/shinka_router/evaluator.py',
        'examples/shinka_router/evolution.py',
        'examples/shinka_router/pareto.py',
        'examples/shinka_router/__init__.py',
        'examples/shinka_router/math_eval.py',
        'examples/shinka_router/AIME_Dataset_1983_2025.csv',
        '.env',
    ]
    
    missing = [f for f in required if not Path(f).exists()]
    
    if missing:
        print("Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False
    return True


def main():
    print("ShinkaRouter AIME Training Setup")
    print("="*50)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("   ✓ All packages installed")
    
    # Check files
    print("\n2. Checking required files...")
    if not check_files():
        sys.exit(1)
    print("   ✓ All files present")
    
    # Check API key
    print("\n3. Checking API key...")
    from dotenv import load_dotenv
    import os
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        print("   ✗ OPENAI_API_KEY not found in .env")
        sys.exit(1)
    print("   ✓ API key configured")
    
    print("\n" + "="*50)
    print("Setup complete! Ready to train.")
    print("\nRun training with:")
    print("  python examples/shinka_router/train_aime.py")
    print("\nConfiguration (edit train_aime.py):")
    print("  NUM_PROBLEMS = 15        # Problems to use")
    print("  NUM_GENERATIONS = 5      # Evolution generations")
    print("  POPULATION_SIZE = 6      # Router population")


if __name__ == "__main__":
    main()