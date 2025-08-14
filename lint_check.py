#!/usr/bin/env python3
"""Script to perform comprehensive linting checks on agent.py"""

import ast
import re

def check_syntax(filename):
    """Check Python syntax"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        ast.parse(content)
        print("✅ Syntax check: PASSED")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax check: FAILED - {e}")
        return False

def check_line_length(filename, max_length=100):
    """Check line length violations"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    violations = []
    for i, line in enumerate(lines, 1):
        if len(line.rstrip()) > max_length:
            violations.append((i, len(line.rstrip()), line.rstrip()))
    
    if violations:
        print(f"❌ Line length check: FAILED - {len(violations)} violations")
        for line_num, length, content in violations:
            print(f"   Line {line_num} ({length} chars): {content[:80]}...")
        return False
    else:
        print("✅ Line length check: PASSED")
        return True

def check_trailing_whitespace(filename):
    """Check for trailing whitespace"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    violations = []
    for i, line in enumerate(lines, 1):
        if line.rstrip() != line.rstrip('\n'):
            violations.append(i)
    
    if violations:
        print(f"❌ Trailing whitespace check: FAILED - {len(violations)} violations")
        print(f"   Lines with trailing whitespace: {violations}")
        return False
    else:
        print("✅ Trailing whitespace check: PASSED")
        return True

def check_file_ending(filename):
    """Check file ends with single newline"""
    with open(filename, 'rb') as f:
        content = f.read()
    
    if content.endswith(b'\n\n'):
        print("❌ File ending check: FAILED - Multiple trailing newlines")
        return False
    elif not content.endswith(b'\n'):
        print("❌ File ending check: FAILED - No trailing newline")
        return False
    else:
        print("✅ File ending check: PASSED")
        return True

def check_imports(filename):
    """Check for unused imports (basic check)"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Basic check for common unused imports
    unused_imports = []
    
    # Check if imports are used
    imports_to_check = [
        ('AnyMessage', 'from langchain_core.messages import.*AnyMessage'),
        ('AIMessage', 'from langchain_core.messages import.*AIMessage'),
        ('add_messages', 'from langgraph.graph.message import add_messages'),
        ('Annotated', 'from typing import.*Annotated'),
        ('TypedDict', 'from typing_extensions import TypedDict')
    ]
    
    for import_name, import_pattern in imports_to_check:
        if re.search(import_pattern, content) and import_name not in content.replace(import_pattern, ''):
            unused_imports.append(import_name)
    
    if unused_imports:
        print(f"❌ Import check: FAILED - Unused imports: {unused_imports}")
        return False
    else:
        print("✅ Import check: PASSED")
        return True

def main():
    filename = 'agent.py'
    print("🔍 Running comprehensive linting checks on agent.py")
    print("=" * 60)
    
    checks = [
        check_syntax(filename),
        check_line_length(filename),
        check_trailing_whitespace(filename),
        check_file_ending(filename),
        check_imports(filename)
    ]
    
    print("=" * 60)
    
    if all(checks):
        print("🎉 ALL LINTING CHECKS PASSED!")
        print("✅ Code quality requirements satisfied")
        return True
    else:
        failed_count = len([c for c in checks if not c])
        print(f"❌ {failed_count} linting checks failed")
        print("❌ Code quality issues need to be resolved")
        return False

if __name__ == "__main__":
    main()
