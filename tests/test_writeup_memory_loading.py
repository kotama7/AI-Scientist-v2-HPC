
import os
import sys
from ai_scientist.perform_writeup import load_idea_text

def test_memory_loading():
    # Use real experiment path that we know has memory
    exp_dir = "/home/users/takanori.kotama/workplace/AI-Scientist-v2-HPC/experiments/2026-01-15_14-55-10_tail_stable_stencil_v2_attempt_0"
    
    print(f"Testing loading from: {exp_dir}")
    if not os.path.exists(exp_dir):
        print("Experiment dir not found, skipping specific path test.")
        return

    text = load_idea_text(exp_dir)
    
    if "# Generated Experiment Memory" in text:
        print("SUCCESS: Memory header found in loaded text.")
        # Check specific content from memory
        if "Tail-Stable Himeno" in text:
             print("SUCCESS: Memory content snippet found.")
        else:
             print("WARNING: Header found but content vague/missing?")
    else:
        print("FAILURE: Memory header NOT found.")
        print("Loaded text length:", len(text))
        print("Snippet:", text[:500])

if __name__ == "__main__":
    test_memory_loading()
