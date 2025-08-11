#!/usr/bin/env python3
"""
OpenAI API Test Script

This script tests the OpenAI API connectivity and functionality.
It checks if the API key is valid and if the API is responding correctly.
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import time

# Load environment variables
load_dotenv()

def test_openai_api():
    """
    Test OpenAI API connectivity and functionality
    """
    print("🔍 Testing OpenAI API...")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return False
    
    print(f"✅ API Key found: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Test 1: List available models
        print("\n📋 Test 1: Listing available models...")
        try:
            models = client.models.list()
            model_names = [model.id for model in models.data]
            print(f"✅ Successfully retrieved {len(model_names)} models")
            
            # Check for common models
            common_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
            available_common = [model for model in common_models if model in model_names]
            if available_common:
                print(f"✅ Common models available: {', '.join(available_common)}")
            else:
                print("⚠️  No common models found in available models")
                
        except Exception as e:
            print(f"❌ Failed to list models: {str(e)}")
            return False
        
        # Test 2: Simple chat completion
        print("\n💬 Test 2: Testing chat completion...")
        try:
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            print(f"Using model: {model}")
            
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Hello, OpenAI API is working!' and nothing else."}
                ],
                max_tokens=50,
                temperature=0.1
            )
            end_time = time.time()
            
            response_text = response.choices[0].message.content.strip()
            response_time = round(end_time - start_time, 2)
            
            print(f"✅ Response received in {response_time}s")
            print(f"📝 Response: {response_text}")
            
            # Check token usage
            if hasattr(response, 'usage'):
                usage = response.usage
                print(f"🔢 Token usage - Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
            
        except Exception as e:
            print(f"❌ Chat completion failed: {str(e)}")
            return False
        
        # Test 3: Test embeddings (if used in the project)
        print("\n🔤 Test 3: Testing embeddings...")
        try:
            embedding_response = client.embeddings.create(
                model="text-embedding-ada-002",
                input="This is a test sentence for embedding."
            )
            
            embedding_vector = embedding_response.data[0].embedding
            print(f"✅ Embedding generated successfully")
            print(f"📊 Embedding dimensions: {len(embedding_vector)}")
            
        except Exception as e:
            print(f"❌ Embedding test failed: {str(e)}")
            print("⚠️  This might be okay if embeddings are not needed for your use case")
        
        print("\n" + "=" * 50)
        print("🎉 OpenAI API is working correctly!")
        print("✅ All tests passed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print("🔧 Possible solutions:")
        print("   1. Check if your API key is valid")
        print("   2. Verify you have sufficient credits in your OpenAI account")
        print("   3. Check your internet connection")
        print("   4. Ensure the API key has the necessary permissions")
        return False

def check_environment():
    """
    Check environment setup
    """
    print("🔧 Checking environment setup...")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ Found {env_file} file")
    else:
        print(f"⚠️  {env_file} file not found")
        print("   Create a .env file with your OPENAI_API_KEY")
    
    # Check required packages
    required_packages = [("openai", "openai"), ("python-dotenv", "dotenv")]
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} is installed")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} is NOT installed")
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 OpenAI API Test Script")
    print("=" * 50)
    
    # Check environment first
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    print("\n")
    
    # Test OpenAI API
    success = test_openai_api()
    
    if success:
        print("\n🎯 RESULT: OpenAI API is working perfectly!")
        sys.exit(0)
    else:
        print("\n💥 RESULT: OpenAI API test failed!")
        sys.exit(1)
