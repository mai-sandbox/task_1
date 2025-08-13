#!/usr/bin/env python3
"""
Simple test script to verify the agent implementation works correctly.
"""

try:
    from agent import app, initialize_state
    from langchain_core.messages import HumanMessage
    
    print("✅ Successfully imported agent components")
    
    # Test state initialization
    test_messages = [HumanMessage("Hello, can you help me with a question?")]
    initial_state = initialize_state(test_messages)
    
    print("✅ Successfully initialized state")
    print(f"Initial state keys: {list(initial_state.keys())}")
    
    # Test that app is compiled
    if app is not None:
        print("✅ Graph compiled successfully and app is available")
        print(f"App type: {type(app)}")
    else:
        print("❌ App is None - compilation failed")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
