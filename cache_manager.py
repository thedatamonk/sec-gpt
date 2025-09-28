#!/usr/bin/env python3
"""
SEC Company Data Cache Manager

Usage:
    python cache_manager.py --info          # Show cache information
    python cache_manager.py --refresh       # Force refresh cache
    python cache_manager.py --clear         # Clear cache
"""

import argparse
import sys
from pathlib import Path
import shutil

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from mods.query_parser import SECQueryParser

def show_cache_info():
    """Display cache information"""
    parser = SECQueryParser()
    info = parser.get_cache_info()
    
    print("=== SEC Company Data Cache Info ===")
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

def refresh_cache():
    """Force refresh the cache"""
    print("Refreshing SEC company data cache...")
    parser = SECQueryParser(force_refresh=True)
    print("Cache refresh completed!")

def clear_cache():
    """Clear the cache directory"""
    parser = SECQueryParser()
    cache_dir = parser.cache_dir
    
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cache directory {cache_dir} cleared successfully!")
    else:
        print("No cache directory found.")

def main():
    parser = argparse.ArgumentParser(description="Manage SEC Company Data Cache")
    parser.add_argument("--info", action="store_true", help="Show cache information")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cache")
    parser.add_argument("--clear", action="store_true", help="Clear cache")
    
    args = parser.parse_args()
    
    if args.info:
        show_cache_info()
    elif args.refresh:
        refresh_cache()
    elif args.clear:
        clear_cache()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()