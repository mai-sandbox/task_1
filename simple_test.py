#!/usr/bin/env python3
"""
Simple test script for the Two-Stage Agent Review Workflow

This script performs basic structural tests without making API calls
to verify the workflow is properly constructed.
"""

import sys
import traceback
from langchain_core.messages import HumanMessage


def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from agent import app, create_review_workflow, create_dummy_react_agent
        print("✅ Successfully imported agent components")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_workflow_structure():
    """Test that the workflow has the expected structure."""
    print("\n🧪 Testing workflow structure...")
    
    try:
        from agent import app
        
        # Check that app is compiled
        if hasattr(app, 'invoke'):
            print("✅ App has invoke method")
        else:
            print("❌ App missing invoke method")
            return False
            
        # Check graph structure
        if hasattr(app, 'get_graph'):
            graph = app.get_graph()
            nodes = list(graph.nodes.keys())
            print(f"✅ Graph nodes: {nodes}")
            
            # Check for expected nodes
            expected_nodes = ['react_agent', 'reviewer', 'add_feedback']
            missing_nodes = [node for node in expected_nodes if node not in nodes]
            if missing_nodes:
                print(f"⚠️  Missing expected nodes: {missing_nodes}")
            else:
                print("✅ All expected nodes present")
                
        else:
            print("⚠️  Cannot inspect graph structure")
            
        return True
        
    except Exception as e:
        print(f"❌ Workflow structure test failed: {e}")
        traceback.print_exc()
        return False


def test_generic_function():
    """Test that the generic create_review_workflow function works."""
    print("\n🧪 Testing generic workflow creation...")
    
    try:
        from agent import create_review_workflow, create_dummy_react_agent
        
        # Create a dummy agent
        dummy_agent = create_dummy_react_agent()
        print("✅ Created dummy React agent")
        
        # Create workflow with the dummy agent
        workflow = create_review_workflow(dummy_agent, max_iterations=2)
        print("✅ Created review workflow with dummy agent")
        
        # Check that workflow is compiled
        if hasattr(workflow, 'invoke'):
            print("✅ Workflow is properly compiled")
        else:
            print("❌ Workflow not properly compiled")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Generic function test failed: {e}")
        traceback.print_exc()
        return False


def test_state_structure():
    """Test that the state structure is correct."""
    print("\n🧪 Testing state structure...")
    
    try:
        # Test basic input structure
        test_input = {
            "messages": [HumanMessage(content="Test message")],
            "iteration_count": 0,
            "original_query": ""
        }
        
        print("✅ Test input structure created successfully")
        print(f"   Messages: {len(test_input['messages'])}")
        print(f"   Iteration count: {test_input['iteration_count']}")
        print(f"   Original query: '{test_input['original_query']}'")
        
        return True
        
    except Exception as e:
        print(f"❌ State structure test failed: {e}")
        traceback.print_exc()
        return False


def run_structural_tests():
    """Run all structural tests without API calls."""
    print("=" * 60)
    print(" TWO-STAGE WORKFLOW STRUCTURAL TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Workflow Structure", test_workflow_structure),
        ("Generic Function", test_generic_function),
        ("State Structure", test_state_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running: {test_name}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print(" TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nStructural Tests: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All structural tests passed!")
        print("📝 The workflow is properly structured and ready for runtime testing.")
        print("💡 Use 'langgraph dev' to test with actual API calls.")
    else:
        print("⚠️  Some structural tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    print("Starting Two-Stage Agent Review Workflow Structural Tests...")
    success = run_structural_tests()
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Set ANTHROPIC_API_KEY environment variable")
        print("2. Run 'langgraph dev' to start the development server")
        print("3. Test the workflow with actual queries")
    
    sys.exit(0 if success else 1)
