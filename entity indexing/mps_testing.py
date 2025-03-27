import torch

def verify_mps_setup():
    print("PyTorch Version:", torch.__version__)
    
    print("\nMPS Setup Information:")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        try:
            mps_device = torch.device("mps")
            x = torch.ones(5, device=mps_device)
            y = torch.ones(5, device=mps_device) * 2
            z = x + y
            
            print("\nTest tensor operation successful:")
            print(f"x (ones): {x}")
            print(f"y (twos): {y}")
            print(f"z (x+y): {z}")
            print("\nMPS setup is working correctly! ✅")
            
        except Exception as e:
            print(f"\nError during MPS testing: {str(e)}")
    else:
        print("\nMPS is not available on this system.")
        print("Requirements for MPS:")
        print("- macOS 12.3 or later")
        print("- PyTorch 1.12 or later")
        print("- Apple Silicon (M1/M2) or AMD GPU")

if __name__ == "__main__":
    verify_mps_setup()