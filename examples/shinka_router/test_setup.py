#!/usr/bin/env python3
"""
Test script for ShinkaRouter setup.

Run this to verify your environment is configured correctly.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages are available."""
    print("Testing imports...")
    
    try:
        import openai
        print("✓ openai")
    except ImportError:
        print("✗ openai - Install with: pip install openai")
        return False
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError:
        print("✗ pandas - Install with: pip install pandas")
        return False
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError:
        print("✗ numpy - Install with: pip install numpy")
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError:
        print("✗ matplotlib - Install with: pip install matplotlib")
        return False
    
    try:
        import seaborn
        print("✓ seaborn")
    except ImportError:
        print("✗ seaborn - Install with: pip install seaborn")
        return False
    
    try:
        from shinka.core import EvolutionRunner
        print("✓ shinka")
    except ImportError:
        print("✗ shinka - Install ShinkaEvolve first")
        return False
    
    return True


def test_env_vars():
    """Test that required environment variables are set."""
    print("\nTesting environment variables...")
    
    import os
    from dotenv import load_dotenv
    
    # Try to load .env
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"✓ Found .env at {env_path}")
    else:
        print(f"⚠ No .env file found at {env_path}")
    
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OPENAI_API_KEY is set")
        return True
    else:
        print("✗ OPENAI_API_KEY not set")
        print("  Set it in .env file or export OPENAI_API_KEY=your_key")
        return False


def test_dataset():
    """Test that AIME dataset is available."""
    print("\nTesting dataset...")
    
    dataset_path = Path(__file__).parent / "AIME_Dataset_1983_2025.csv"
    
    if dataset_path.exists():
        print(f"✓ Found AIME dataset at {dataset_path}")
        
        # Try to load it
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            print(f"  Dataset has {len(df)} problems")
            years = sorted(df['Year'].unique())
            print(f"  Years available: {years[0]}-{years[-1]}")
            return True
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return False
    else:
        print(f"✗ AIME dataset not found at {dataset_path}")
        print("  Copy from examples/adas_aime/ or download separately")
        return False


def test_query_llm():
    """Test that LLM query works."""
    print("\nTesting LLM query...")
    
    try:
        from utils import query_llm
        
        response, cost = query_llm(
            prompt="Say 'test successful' and nothing else.",
            system="You are a helpful assistant.",
            temperature=0.0,
            model_name="gpt-4o-mini"
        )
        
        if response and cost >= 0:
            print(f"✓ LLM query successful")
            print(f"  Response: {response[:50]}...")
            print(f"  Cost: ${cost:.6f}")
            return True
        else:
            print("✗ LLM query failed")
            return False
            
    except Exception as e:
        print(f"✗ LLM query error: {e}")
        return False


def test_agent_instantiation():
    """Test that Agent can be instantiated."""
    print("\nTesting Agent instantiation...")
    
    try:
        from initial import Agent
        from utils import query_llm
        from functools import partial
        
        base_query = partial(query_llm, model_name="gpt-4o-mini")
        agent = Agent(base_query)
        
        print("✓ Agent instantiated successfully")
        print(f"  Primitives available: quick_solve, deep_think, verify, "
              f"python_calc, ensemble_vote, self_critique, estimate_difficulty")
        return True
        
    except Exception as e:
        print(f"✗ Agent instantiation error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("ShinkaRouter Setup Test")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Environment Variables", test_env_vars),
        ("Dataset", test_dataset),
        ("Agent", test_agent_instantiation),
    ]
    
    # Run LLM test last (costs money)
    print("\n" + "=" * 80)
    response = input("Run LLM query test? (costs ~$0.0001) [y/N]: ")
    if response.lower() == 'y':
        tests.append(("LLM Query", test_query_llm))
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run ShinkaRouter.")
        print("\nNext steps:")
        print("  1. python run_evo.py  # Start evolution")
        print("  2. shinka_visualize --port 8888 --open  # Monitor progress")
        print("  3. python analyze_router.py  # Analyze results")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())