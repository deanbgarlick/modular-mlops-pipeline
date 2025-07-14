#!/usr/bin/env python3
"""ML Training VM Deployment Script in Python"""

import os
import dotenv
import sys
import time
import subprocess
from dataclasses import dataclass
from typing import Optional
from google.cloud import compute_v1

dotenv.load_dotenv()

@dataclass
class VMConfig:
    project_id: str = os.getenv("GCLOUD_PROJECT_ID", "your-project-id")
    zone: str = "us-central1-a" 
    repo_url: str = "https://github.com/yourusername/repo.git"
    branch: str = "main"
    machine_type: str = "n1-standard-1"
    preemptible: bool = True
    auto_shutdown: bool = True
    service_account: str = os.getenv("ML_SERVICE_ACCOUNT", "")
    # New field for ML pipeline mode
    ml_mode: str = "single"  # "single", "suite", or "dispatcher"

class MLTrainingDeployer:
    def __init__(self, config: VMConfig):
        self.config = config
        self.compute_client = compute_v1.InstancesClient()
        
    def deploy_vm(self, vm_name: Optional[str] = None) -> str:
        """Deploy VM and return instance name"""
        if not vm_name:
            vm_name = f"ml-training-{self.config.ml_mode}-{int(time.time())}"
            
        # Read startup script
        with open('startup-script-local.sh', 'r') as f:
            startup_script = f.read()
            
        # Configure instance
        instance_config = {
            "name": vm_name,
            "machine_type": f"zones/{self.config.zone}/machineTypes/{self.config.machine_type}",
            "scheduling": {"preemptible": self.config.preemptible},
            "disks": [{
                "boot": True,
                "auto_delete": True,
                "initialize_params": {
                    "source_image": "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts",
                    "disk_size_gb": 50,
                    "disk_type": f"zones/{self.config.zone}/diskTypes/pd-ssd"
                }
            }],
            "network_interfaces": [{
                "network": f"projects/{self.config.project_id}/global/networks/default"
            }],
            "metadata": {
                "items": [
                    {"key": "startup-script", "value": startup_script},
                    {"key": "repo-url", "value": self.config.repo_url},
                    {"key": "branch", "value": self.config.branch},
                    {"key": "auto-shutdown", "value": str(self.config.auto_shutdown).lower()},
                    {"key": "ml-mode", "value": self.config.ml_mode}
                ]
            },
            "service_accounts": [{
                "email": self.config.service_account if self.config.service_account else "default",
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
            }],
            "tags": {"items": ["ml-training"]}
        }
        
        # Create instance
        instance = compute_v1.Instance(instance_config)
        operation = self.compute_client.insert(
            project=self.config.project_id,
            zone=self.config.zone,
            instance_resource=instance
        )
        
        print(f"✓ VM {vm_name} deployed successfully!")
        print(f"🎯 ML Mode: {self.config.ml_mode}")
        
        # Show mode-specific info
        if self.config.ml_mode == "single":
            print(f"📄 Running: python main_single_run.py")
        elif self.config.ml_mode == "suite":
            print(f"📊 Running: python main_experiment_run.py")
        else:
            print(f"🚀 Running: python main.py (dispatcher mode)")
        
        print(f"📊 Logs will be saved to: gs://{self.config.project_id}-ml-logs/{vm_name}_<timestamp>/")
        print(f"🔄 Stream logs: python deploy.py --action stream --name {vm_name}")
        print(f"📋 Monitor VM: python deploy.py --action monitor --name {vm_name}")
        print(f"📜 Get logs: ./get_logs.sh {vm_name}")
        return vm_name
    
    def get_logs(self, vm_name: str) -> str:
        """Get VM serial console output"""
        result = subprocess.run([
            "gcloud", "compute", "instances", "get-serial-port-output", 
            vm_name, f"--zone={self.config.zone}"
        ], capture_output=True, text=True)
        return result.stdout
    
    def monitor_vm(self, vm_name: str):
        """Monitor VM until completion"""
        while True:
            try:
                instance = self.compute_client.get(
                    project=self.config.project_id,
                    zone=self.config.zone,
                    instance=vm_name
                )
                status = instance.status
                print(f"VM Status: {status} ({time.strftime('%H:%M:%S')})")
                
                if status == "TERMINATED":
                    print("✓ VM has terminated")
                    break
                    
                time.sleep(30)
                
            except Exception as e:
                print(f"✗ VM not found: {e}")
                break
    
    def stream_logs(self, vm_name: str):
        """Stream VM logs in real-time"""
        print(f"🔄 Streaming logs from {vm_name}... (Ctrl+C to stop)")
        try:
            subprocess.run([
                "gcloud", "compute", "instances", "tail-serial-port-output",
                vm_name, f"--zone={self.config.zone}"
            ])
        except KeyboardInterrupt:
            print("\n📋 Log streaming stopped")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy ML training to GCP VM')
    parser.add_argument('--action', choices=['deploy', 'logs', 'monitor', 'stream'], default='deploy', help='Action to perform (default: deploy)')
    parser.add_argument('--name', default=None, help='VM name (auto-generated if not provided)')
    parser.add_argument('--machine-type', default='n1-standard-1', help='Machine type (default: n1-standard-1)')
    parser.add_argument('--service-account', default=None, help='Service account email (default: uses environment ML_SERVICE_ACCOUNT or VM default)')
    parser.add_argument('--no-preemptible', action='store_true', help='Disable preemptible instances (default: False)')
    parser.add_argument('--no-auto-shutdown', action='store_true', help='Disable auto-shutdown (default: False)')
    # New argument for ML pipeline mode
    parser.add_argument('--ml-mode', choices=['single', 'suite', 'dispatcher'], default='single', 
                       help='ML pipeline mode: single experiment, experiment suite, or dispatcher (default: single)')
    
    args = parser.parse_args()
    
    config = VMConfig(
        machine_type=args.machine_type,
        preemptible=not args.no_preemptible,
        auto_shutdown=not args.no_auto_shutdown,
        service_account=getattr(args, 'service_account', None) or os.getenv("ML_SERVICE_ACCOUNT", ""),
        ml_mode=args.ml_mode
    )
    
    deployer = MLTrainingDeployer(config)
    
    if args.action == 'deploy':
        vm_name = deployer.deploy_vm(args.name)
        print(f"Monitor with: python deploy.py --action monitor --name {vm_name}")
        
        # Show mode-specific usage examples
        print(f"\n🚀 Deployment Examples:")
        print(f"  Single experiment:    python deploy.py --ml-mode single")
        print(f"  Experiment suite:     python deploy.py --ml-mode suite") 
        print(f"  Dispatcher (default): python deploy.py --ml-mode dispatcher")
        
    elif args.action == 'logs':
        if not args.name:
            print("--name required for logs")
            sys.exit(1)
        logs = deployer.get_logs(args.name)
        print(logs)
        
    elif args.action == 'monitor':
        if not args.name:
            print("--name required for monitor")
            sys.exit(1)
        deployer.monitor_vm(args.name)
        
    elif args.action == 'stream':
        if not args.name:
            print("--name required for stream")
            sys.exit(1)
        deployer.stream_logs(args.name)

if __name__ == "__main__":
    main() 