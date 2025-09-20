#!/usr/bin/env python3
"""
Auto-configuration for Blueberry LLM
Detects hardware and automatically configures optimal training setup
"""

import platform
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

@dataclass
class AutoConfig:
    """Auto-detected configuration"""
    # Hardware
    num_gpus: int
    gpu_memory_gb: float
    
    # Model (auto-scaled)
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    num_experts: int
    
    # Training (auto-optimized)
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    max_seq_len: int
    
    # Performance
    use_distributed: bool
    use_amp: bool

class BlueberryAutoConfigurator:
    """One class that does everything"""
    
    def __init__(self):
        self.config = self._detect_and_configure()
    
    def _detect_and_configure(self) -> AutoConfig:
        """Main auto-configuration logic"""
        
        # Detect hardware
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if num_gpus == 0:
            return self._cpu_config()
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        total_memory = gpu_memory_gb * num_gpus
        
        # Scale model based on total available memory
        if total_memory < 16:  # Small setup
            config = {
                'd_model': 256, 'n_layers': 4, 'n_heads': 4, 'd_ff': 1024,
                'num_experts': 4, 'batch_size': 8, 'max_seq_len': 512
            }
        elif total_memory < 64:  # Medium setup
            config = {
                'd_model': 384, 'n_layers': 6, 'n_heads': 8, 'd_ff': 1536,
                'num_experts': 8, 'batch_size': 16, 'max_seq_len': 1024
            }
        elif total_memory < 256:  # Large setup
            config = {
                'd_model': 768, 'n_layers': 12, 'n_heads': 12, 'd_ff': 3072,
                'num_experts': 16, 'batch_size': 32, 'max_seq_len': 2048
            }
        else:  # Massive setup
            config = {
                'd_model': 1536, 'n_layers': 24, 'n_heads': 24, 'd_ff': 6144,
                'num_experts': 32, 'batch_size': 64, 'max_seq_len': 4096
            }
        
        # Adjust for limited memory per GPU
        if gpu_memory_gb < 12:
            config['batch_size'] = max(1, config['batch_size'] // 2)
        
        # Set training parameters
        gradient_accumulation_steps = max(1, 32 // config['batch_size'])
        
        return AutoConfig(
            num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb,
            **config,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=5000,
            learning_rate=0.01,
            use_distributed=(num_gpus > 1),
            use_amp=True
        )
    
    def _cpu_config(self) -> AutoConfig:
        """Minimal config for CPU-only systems"""
        return AutoConfig(
            num_gpus=0, gpu_memory_gb=0,
            d_model=128, n_layers=2, n_heads=4, d_ff=512, num_experts=2,
            batch_size=4, gradient_accumulation_steps=8, max_steps=1000,
            learning_rate=0.001, max_seq_len=256,
            use_distributed=False, use_amp=False
        )
    
    def _format_memory(self, value_gb: float) -> str:
        return f"{value_gb:.1f} GB"

    def print_config(self):
        """Print detected configuration"""
        print("🫐 Blueberry LLM Auto-Configuration")
        print("=" * 50)

        if self.config.num_gpus == 0:
            print("🖥️  Mode: CPU Training (Limited)")
        else:
            print(f"🚀 Mode: GPU Training ({self.config.num_gpus} GPUs)")
            print(f"   Memory: {self.config.gpu_memory_gb:.1f} GB per GPU")

        # Hardware summary
        cpu_name = platform.processor() or "Unknown CPU"
        machine = platform.machine()
        print(f"🧮 CPU: {cpu_name} ({machine})")

        if psutil is not None:
            try:
                total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
                print(f"🗄️  System RAM: {self._format_memory(total_ram_gb)}")
            except Exception:
                pass
        else:
            print("🗄️  System RAM: psutil not installed (skip)")

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                cc = f"{props.major}.{props.minor}"
                print(
                    f"🎮 GPU #{idx}: {props.name} | {props.total_memory / (1024 ** 3):.1f} GB | SM {cc} | {props.multi_processor_count} SMs"
                )
        else:
            print("🎮 GPU: none detected")

        print(f"📏 Model: {self.config.d_model}d × {self.config.n_layers}L × {self.config.n_heads}H")
        print(f"🧠 Experts: {self.config.num_experts}")
        print(f"📊 Batch: {self.config.batch_size} (accum: {self.config.gradient_accumulation_steps})")
        print(f"📝 Sequence: {self.config.max_seq_len}")
        print(f"⚡ Mixed Precision: {'Yes' if self.config.use_amp else 'No'}")
        
        if self.config.use_distributed:
            print(f"🌐 Data Parallel: Yes (across {self.config.num_gpus} GPUs)")
            print(f"   Run with: torchrun --nproc_per_node={self.config.num_gpus} train_auto.py")
        
        print("=" * 50)
    
    def get_model_config(self):
        """Convert to MoEModelConfig format"""
        from llm import MoEModelConfig
        
        return MoEModelConfig(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            batch_size=self.config.batch_size,
            max_steps=self.config.max_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            muon_lr=self.config.learning_rate,
            learning_rate=self.config.learning_rate,
            optimizer="muon",
            max_seq_len=self.config.max_seq_len,
            num_experts=self.config.num_experts,
            use_amp=self.config.use_amp,
        )

def auto_configure() -> BlueberryAutoConfigurator:
    """One function call to auto-configure everything"""
    return BlueberryAutoConfigurator()

if __name__ == "__main__":
    # Demo the auto-configuration
    configurator = auto_configure()
    configurator.print_config()
