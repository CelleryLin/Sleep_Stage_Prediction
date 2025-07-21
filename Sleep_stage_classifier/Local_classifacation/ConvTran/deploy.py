"""
Deploy ConvTran model to ONNX format for inference.
"""

import os
import sys
import torch
import torch.onnx
import numpy as np
import onnxruntime as ort
from Dataset.ECGDataset import ECGDataset
from Models.model import ConvTran_timepos

sys.path.append(os.path.join(os.getcwd(), '../../..'))


def export_to_onnx(model, model_path, onnx_path, input_shape=(1, 2, 150), device='cpu'):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        model_path: Path to saved PyTorch model weights
        onnx_path: Path to save ONNX model
        input_shape: Input shape for the model (batch_size, channels, sequence_length)
        device: Device to run the model on
    """
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dummy inputs matching the expected input format
    # ECG data input
    dummy_x = torch.randn(input_shape, dtype=torch.float32, device=device)
    # Time position input (assuming same sequence length as ECG)
    dummy_time_pos = torch.randn(input_shape[0], input_shape[2], dtype=torch.float32, device=device)
    
    print(f"Dummy input shapes - X: {dummy_x.shape}, time_pos: {dummy_time_pos.shape}")
    
    # Test the model with dummy inputs
    with torch.no_grad():
        output = model(dummy_x, dummy_time_pos)
        print(f"Model output shape: {output.shape}")
    
    # Export to ONNX
    print(f"Exporting model to ONNX format: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_x, dummy_time_pos),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['ecg_input', 'time_pos_input'],
        output_names=['prediction'],
        dynamic_axes={
            'ecg_input': {0: 'batch_size'},
            'time_pos_input': {0: 'batch_size'},
            'prediction': {0: 'batch_size'}
        }
    )
    
    print(f"Model successfully exported to: {onnx_path}")
    return onnx_path


def verify_onnx_model(onnx_path, original_model, model_path, input_shape=(1, 2, 150), device='cpu', tolerance=1e-5):
    """
    Verify that ONNX model produces same outputs as original PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        original_model: Original PyTorch model
        model_path: Path to PyTorch model weights
        input_shape: Input shape for testing
        device: Device to run PyTorch model on
        tolerance: Tolerance for numerical differences
    """
    print("\nVerifying ONNX model...")
    
    # Load original PyTorch model
    original_model.load_state_dict(torch.load(model_path, map_location=device))
    original_model.eval()
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test inputs
    test_x = torch.randn(input_shape, dtype=torch.float32, device=device)
    test_time_pos = torch.randn(input_shape[0], input_shape[2], dtype=torch.float32, device=device)
    
    # Get PyTorch model output
    with torch.no_grad():
        pytorch_output = original_model(test_x, test_time_pos).cpu().numpy()
    
    # Get ONNX model output
    onnx_inputs = {
        'ecg_input': test_x.cpu().numpy(),
        'time_pos_input': test_time_pos.cpu().numpy()
    }
    onnx_output = ort_session.run(None, onnx_inputs)[0]
    
    # Compare outputs
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    print(f"Maximum difference between PyTorch and ONNX outputs: {max_diff}")
    
    if max_diff < tolerance:
        print("✓ ONNX model verification successful!")
        return True
    else:
        print(f"✗ ONNX model verification failed! Difference exceeds tolerance ({tolerance})")
        return False


def optimize_onnx_model(onnx_path, optimized_path=None):
    """
    Optimize ONNX model for inference.
    
    Args:
        onnx_path: Path to original ONNX model
        optimized_path: Path to save optimized model (optional)
    """
    try:
        import onnx
        from onnxruntime.tools import optimizer
        
        if optimized_path is None:
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        print(f"Optimizing ONNX model...")
        
        # Load and optimize model
        model = onnx.load(onnx_path)
        optimized_model = optimizer.optimize_model(
            onnx_path,
            model_type='bert',  # Use general optimization
            num_heads=8,  # Match your model configuration
            hidden_size=16  # Match your embedding size
        )
        
        onnx.save(optimized_model, optimized_path)
        print(f"Optimized model saved to: {optimized_path}")
        return optimized_path
        
    except ImportError:
        print("ONNX optimization tools not available. Install with: pip install onnxruntime-tools")
        return onnx_path


class ONNXInference:
    """
    Wrapper class for ONNX model inference.
    """
    
    def __init__(self, onnx_path, providers=None):
        """
        Initialize ONNX inference session.
        
        Args:
            onnx_path: Path to ONNX model
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
            
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"ONNX model loaded with providers: {self.session.get_providers()}")
        print(f"Input names: {self.input_names}")
        print(f"Output names: {self.output_names}")
    
    def predict(self, ecg_data, time_pos_data, threshold=0.5):
        """
        Run inference on input data.
        
        Args:
            ecg_data: ECG input data (numpy array)
            time_pos_data: Time position input data (numpy array)
            threshold: Classification threshold
            
        Returns:
            predictions: Binary predictions
            probabilities: Raw prediction probabilities
        """
        # Ensure inputs are numpy arrays with correct dtype
        if isinstance(ecg_data, torch.Tensor):
            ecg_data = ecg_data.cpu().numpy()
        if isinstance(time_pos_data, torch.Tensor):
            time_pos_data = time_pos_data.cpu().numpy()
            
        ecg_data = ecg_data.astype(np.float32)
        time_pos_data = time_pos_data.astype(np.float32)
        
        # Run inference
        inputs = {
            self.input_names[0]: ecg_data,
            self.input_names[1]: time_pos_data
        }
        
        outputs = self.session.run(self.output_names, inputs)
        probabilities = outputs[0].squeeze()
        
        # Apply threshold for binary classification
        predictions = (probabilities > threshold).astype(int)
        
        return predictions, probabilities
    
    def benchmark(self, input_shape=(1, 2, 150), num_runs=100):
        """
        Benchmark inference speed.
        
        Args:
            input_shape: Shape of input data
            num_runs: Number of inference runs for benchmarking
        """
        import time
        
        # Create dummy data
        ecg_data = np.random.randn(*input_shape).astype(np.float32)
        time_pos_data = np.random.randn(input_shape[0], input_shape[2]).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            self.predict(ecg_data, time_pos_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            self.predict(ecg_data, time_pos_data)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Throughput: {1000/avg_time:.2f} samples/second")


def deploy_model(model_path, config, output_dir='./deployment', optimize=True):
    """
    Complete deployment pipeline: export to ONNX, verify, and optimize.
    
    Args:
        model_path: Path to trained PyTorch model
        config: Model configuration dictionary
        output_dir: Directory to save deployment files
        optimize: Whether to optimize the ONNX model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    device = torch.device('cpu')  # Use CPU for deployment
    model = ConvTran_timepos(config, num_classes=1).to(device)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, 'convtran_model.onnx')
    export_to_onnx(model, model_path, onnx_path, device=device)
    
    # Verify ONNX model
    verification_success = verify_onnx_model(onnx_path, model, model_path, device=device)
    
    if not verification_success:
        print("Warning: ONNX model verification failed!")
        return None
    
    # Optimize model (optional)
    if optimize:
        optimized_path = optimize_onnx_model(onnx_path)
        final_model_path = optimized_path
    else:
        final_model_path = onnx_path
    
    # Create inference wrapper
    print("\nTesting ONNX inference...")
    inference_engine = ONNXInference(final_model_path)
    
    # Benchmark performance
    print("\nBenchmarking inference performance...")
    inference_engine.benchmark()
    
    # Save deployment info
    deployment_info = {
        'pytorch_model_path': model_path,
        'onnx_model_path': final_model_path,
        'config': config,
        'verification_passed': verification_success,
        'optimized': optimize
    }
    
    np.savez(os.path.join(output_dir, 'deployment_info.npz'), **deployment_info)
    
    print(f"\nDeployment completed successfully!")
    print(f"ONNX model saved to: {final_model_path}")
    print(f"Deployment info saved to: {os.path.join(output_dir, 'deployment_info.npz')}")
    
    return inference_engine


if __name__ == "__main__":
    # Model configuration (should match training configuration)
    config = {
        'max_len': 150,
        'Data_shape': (0, 2, 150),
        'emb_size': 16,
        'num_heads': 8,
        'dim_ff': 256,
        'Fix_pos_encode': 'tAPE',
        'Rel_pos_encode': 'eRPE',
        'dropout': 0.01,
    }
    
    # Deployment parameters
    model_path = './output/20241125165230/model.pth'  # Update with your model path
    output_dir = './deployment'
    
    # Deploy the model
    inference_engine = deploy_model(
        model_path=model_path,
        config=config,
        output_dir=output_dir,
        optimize=True
    )
    
    if inference_engine:
        print("\n" + "="*50)
        print("DEPLOYMENT SUCCESSFUL")
        print("="*50)
        print("Your model is ready for production use!")
        print(f"Use the ONNX model at: {output_dir}/convtran_model.onnx")
        print("\nExample usage:")
        print("from deploy import ONNXInference")
        print("inference = ONNXInference('deployment/convtran_model.onnx')")
        print("predictions, probabilities = inference.predict(ecg_data, time_pos_data)")
    else:
        print("Deployment failed. Please check the error messages above.")
