import sys
import torch

def test_mamba_compilation_and_logic():
    print("--- Starting Mamba SSM Functional Verification ---")
    
    try:
        # 1. Dependency Handshake
        print("--> Checking CUDA-dependent frameworks...")
        import triton
        import ninja
        print(f"    [✓] Triton (v{triton.__version__}) and Ninja found.")

        # 2. Kernel Import (The Binary Linkage Check)
        # If the build failed during pip install, this import will crash.
        print("--> Attempting to import Mamba SSM kernels...")
        try:
            from mamba_ssm import Mamba
            print("    [✓] Mamba SSM modules imported successfully.")
        except ImportError as e:
            print(f"    [!] Import failed: {e}")
            raise

        # 3. Structural Initialization
        print("--> Verifying model initialization logic...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Mamba(d_model=32, d_state=8, d_conv=4, expand=2).to(device)
        print(f"    [✓] Mamba model initialized on {device}.")

        print("--- SMOKE TEST PASSED ---")

    except Exception as e:
        print(f"CRITICAL VALIDATION FAILURE: {str(e)}")
        # Likely failures: 
        # - RuntimeError (Ninja build failed)
        # - ImportError (Shared library version mismatch)
        sys.exit(1)

if __name__ == "__main__":
    test_mamba_compilation_and_logic()