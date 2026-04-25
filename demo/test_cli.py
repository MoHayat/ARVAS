#!/usr/bin/env python3
"""
Quick end-to-end test of the CLI demo.
Simulates a short conversation and verifies the demo runs without errors.
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Patch the demo's console.input to feed pre-programmed messages
from unittest.mock import MagicMock
import cli_demo

# Mock console.input
inputs = [
    "Hi, how are you?",
    "You're useless.",
    "/status",
    "/reset",
    "/quit",
]
input_iter = iter(inputs)

cli_demo.console.input = lambda prompt="": next(input_iter)

# Also mock console.status to avoid spinner issues in non-TTY
class MockStatus:
    def __enter__(self): return self
    def __exit__(self, *args): pass
cli_demo.console.status = lambda *args, **kwargs: MockStatus()

# Suppress print_state_panel to keep output clean
original_print_state = cli_demo.print_state_panel
def quiet_print_state(*args, **kwargs):
    pass
cli_demo.print_state_panel = quiet_print_state

# Run
try:
    cli_demo.main()
    print("\n✅ CLI demo end-to-end test PASSED")
except StopIteration:
    print("\n✅ CLI demo end-to-end test PASSED (ran out of inputs)")
except Exception as e:
    print(f"\n❌ CLI demo test FAILED: {e}")
    import traceback
    traceback.print_exc()
