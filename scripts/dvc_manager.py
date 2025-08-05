#!/usr/bin/env python3
"""
DVC Pipeline Management Script
Usage: python scripts/dvc_manager.py [command]
Commands: init, run, status, metrics, push, pull
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and print the result."""
    print(f"\n{description}")
    print("-" * 50)
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def init_dvc():
    """Initialize DVC."""
    return run_command("dvc init --no-scm", "Initializing DVC...")

def run_pipeline():
    """Run the DVC pipeline."""
    success = run_command("dvc repro", "Running DVC pipeline...")
    if success:
        run_command("dvc metrics show", "Showing metrics...")
    return success

def show_status():
    """Show DVC status."""
    return run_command("dvc status", "Showing DVC status...")

def show_metrics():
    """Show DVC metrics."""
    return run_command("dvc metrics show", "Showing DVC metrics...")

def push_data():
    """Push data to remote."""
    return run_command("dvc push", "Pushing data to remote...")

def pull_data():
    """Pull data from remote."""
    return run_command("dvc pull", "Pulling data from remote...")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/dvc_manager.py [command]")
        print("Commands: init, run, status, metrics, push, pull")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    if command == "init":
        init_dvc()
    elif command == "run":
        run_pipeline()
    elif command == "status":
        show_status()
    elif command == "metrics":
        show_metrics()
    elif command == "push":
        push_data()
    elif command == "pull":
        pull_data()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: init, run, status, metrics, push, pull")
        sys.exit(1)

if __name__ == "__main__":
    main()
