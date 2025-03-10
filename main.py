"""
Anus - Autonomous Networked Utility System
Main entry point for the Anus AI agent framework
"""

import argparse
import sys
import os
import yaml
import os
import yaml
from anus.core.orchestrator import AgentOrchestrator
from anus.ui.cli import CLI

def main():
    """Main entry point for the Anus AI agent"""
    parser = argparse.ArgumentParser(description="Anus AI - Autonomous Networked Utility System")
    
    # Add command as first positional argument
    parser.add_argument("command", nargs="?", default="interactive", 
                      help="Command to execute (init, run, interactive)")
    
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multi"], help="Agent mode")
    parser.add_argument("--task", type=str, help="Task description")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Handle the init command
    if args.command == "init":
        create_config()
        return
    
    # Initialize the CLI
    cli = CLI(verbose=args.verbose)
    
    # Display welcome message
    cli.display_welcome()
    
    # Initialize the agent orchestrator
    orchestrator = AgentOrchestrator(config_path=args.config)
    
    # If task is provided as argument, execute it
    if args.task or args.command == "run":
        task = args.task if args.task else args.command
        result = orchestrator.execute_task(task, mode=args.mode)
        cli.display_result(result)
        return
    
    # Otherwise, start interactive mode
    cli.start_interactive_mode(orchestrator)

def create_config():
    """Create default configuration file"""
    config_dir = os.path.expanduser(".anus")
    os.makedirs(config_dir, exist_ok=True)
    
    config_path = os.path.join(config_dir, "config.yaml")
    
    if os.path.exists(config_path):
        confirm = input(f"Config file already exists at {config_path}. Overwrite? (y/n): ")
        if confirm.lower() != "y":
            print("Aborted.")
            return
    
    default_config = {
        "llm": {
            "provider": "openai",
            "api_key": "your_openai_api_key",
            "model": "gpt-4o"
        },
        "memory": {
            "type": "hybrid",
            "persistence": False,
            "storage_path": None
        },
        "tools": {
            "browser": {"headless": True},
            "code": {"sandbox": True}
        }
    }
    
    with open(config_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Configuration initialized at {config_path}")
    print("Please edit this file to add your API keys.")

if __name__ == "__main__":
    main()