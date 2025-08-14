#!/usr/bin/env python3
"""Script to fix file ending issues"""

# Read the file
with open('agent.py', 'r') as f:
    content = f.read()

# Remove all trailing whitespace and newlines, then add exactly one newline
content = content.rstrip() + '\n'

# Write back to file
with open('agent.py', 'w') as f:
    f.write(content)

print("Fixed file ending issues in agent.py")
