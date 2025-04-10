#!/usr/bin/env python3
"""
Test script for OpenVRSystem C++ implementation
This script verifies that the C++ OpenVRSystem class is properly bound to Python
and can initialize the VR system and render frames.
"""

import sys
import os
import time
import traceback

# Add PyTorch library path to LD_LIBRARY_PATH to find libc10.so and other dependencies
try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    print(f"Adding PyTorch library path: {torch_lib_path}")
    
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = torch_lib_path
        
    # We need to restart the script for the environment change to take effect
    if not os.environ.get('OPENVR_TEST_RESTARTED'):
        print("Restarting script with updated LD_LIBRARY_PATH...")
        os.environ['OPENVR_TEST_RESTARTED'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)
except ImportError:
    print("Warning: PyTorch not found, continuing without library path adjustment")

try:
    # Import the OpenVRSystem from the compiled extension
    from optix_resource_system import OpenVRSystem
    print("Successfully imported OpenVRSystem")
except ImportError as e:
    print(f"Error importing OpenVRSystem: {e}")
    print("Make sure the extension is properly compiled with OpenVR support.")
    sys.exit(1)

def main():
    print("=" * 80)
    print("OpenVRSystem C++ Implementation Test")
    print("=" * 80)
    
    # Create the OpenVRSystem instance
    vr_system = OpenVRSystem()
    print("Created OpenVRSystem instance")
    
    try:
        # Initialize the VR system
        print("Initializing OpenVR system...")
        if not vr_system.Initialize():
            print("Failed to initialize OpenVR system")
            return
        
        print("OpenVR system initialized successfully")
        
        # Main loop
        frame_count = 0
        max_frames = 300  # Run for about 5 seconds at 60 fps
        
        print("Rendering frames. Press Ctrl+C to exit.")
        
        start_time = time.time()
        while not vr_system.ShouldClose():
            # Process events
            vr_system.PollEvents()
            
            # Render a frame
            print(f"Rendering frame {frame_count}...")
            vr_system.RenderFrame()
            
            # Update frame counter
            frame_count += 1
            
            # Small delay to avoid maxing out CPU
            #time.sleep(1/60)
        
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        
        print(f"Rendered {frame_count} frames in {elapsed_time:.2f} seconds ({fps:.1f} FPS)")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        print("Shutting down OpenVR system...")
        vr_system.Shutdown()
        print("OpenVR system shutdown complete")

if __name__ == "__main__":
    main()