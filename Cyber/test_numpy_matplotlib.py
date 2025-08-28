#!/usr/bin/env python3
"""
Test script to verify numpy and matplotlib work correctly after reinstallation.
"""

import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
    
    # Test basic numpy operations
    arr = np.array([1, 2, 3, 4, 5])
    print(f"✅ NumPy array creation and operations work: {arr.mean()}")
    
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print(f"✅ Matplotlib imported successfully")
    
    # Test basic matplotlib operations
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    print("✅ Matplotlib plotting works")
    plt.close(fig)
    
except ImportError as e:
    print(f"❌ Matplotlib import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"✅ Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

try:
    import plotly
    print(f"✅ Plotly version: {plotly.__version__}")
except ImportError as e:
    print(f"❌ Plotly import failed: {e}")

print("\n🎉 All critical packages imported successfully!")
print("You can now run your app.py without numpy import errors.")
