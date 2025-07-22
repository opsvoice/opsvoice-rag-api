#!/usr/bin/env python3
"""
Local development script for OpsVoice backend
This script sets up the environment and runs the Flask app locally
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def setup_environment():
    """Set up environment variables for local development"""
    env_vars = {
        'FLASK_ENV': 'development',
        'PORT': '10000',
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', 'your-openai-key-here'),
        'ELEVENLABS_API_KEY': os.getenv('ELEVENLABS_API_KEY', 'your-elevenlabs-key-here'),
        'SECRET_KEY': 'dev-secret-key-change-in-production'
    }
    
    print("🔧 Setting up environment variables...")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   {key}: {'*' * len(value) if 'KEY' in key else value}")
    
    return env_vars

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'flask',
        'langchain',
        'langchain_openai',
        'langchain_community',
        'chromadb',
        'faster_whisper',
        'requests',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'data',
        'data/sop-files',
        'data/chroma_db',
        'data/audio_cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

def run_backend():
    """Run the Flask backend"""
    print("\n🚀 Starting OpsVoice backend...")
    print("   URL: http://localhost:10000")
    print("   Health check: http://localhost:10000/healthz")
    print("   Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Import and run the app
        from app import app, initialize_application
        
        # Initialize the application
        initialize_application()
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=10000,
            debug=True,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting backend: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🎯 OpsVoice Local Development Setup")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return False
    
    # Create directories
    create_directories()
    
    # Run backend
    return run_backend()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 