"""
Preprocess data functions
"""

def preprocess_content(content, which="content"):
    if which == "content":
        return content
    elif which == "label":
        return content
    else:
        raise ValueError(f"Invalid which: {which}")
