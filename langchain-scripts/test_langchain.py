#!/usr/bin/env python3
"""
Basic test script for LangChain functionality
"""

import sys
import os
from langsmith import traceable
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

@traceable
def test_langchain_import():
    """Test if LangChain can be imported successfully"""
    try:
        import langchain
        print(f"✅ LangChain imported successfully! Version: {langchain.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import LangChain: {e}")
        return False

@traceable
def test_basic_components():
    """Test basic LangChain components"""
    try:
        from langchain_community.llms import OpenAI
        from langchain_community.chat_models import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain.chains import LLMChain
        print("✅ Basic LangChain components imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Failed to import basic components: {e}")
        return False

@traceable
def test_simple_chain():
    """Test creating a simple chain"""
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_community.llms import OpenAI
        
        
        # Create a simple prompt template
        template = "What is the capital of {country}?"
        prompt = PromptTemplate(
            input_variables=["country"],
            template=template,
        )
        print("✅ Prompt template created successfully!")
        
        # Note: This would require an actual OpenAI API key to run
        print("ℹ️  To test with actual LLM, you would need an OpenAI API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create simple chain: {e}")
        return False

@traceable
def test_openai_llm():
    """Test OpenAI LLM functionality"""
    try:
        from langchain_community.llms import OpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain.chains import LLMChain
        
        # Create an instance of the OpenAI model
        # Get OpenAI credentials
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("⚠️  No OpenAI API key found. Skipping LLM test.")
            return True
        
        llm = OpenAI(openai_api_key=api_key)
        # Test direct LLM call
        print("🔄 Testing direct LLM call...")
        response = llm("What is LangChain in one sentence?")
        print(f"✅ Direct LLM response: {response[:100]}...")
        
        # Test with prompt template
        print("🔄 Testing LLM with prompt template...")
        template = "What is the capital of {country}?"
        prompt = PromptTemplate(
            input_variables=["country"],
            template=template,
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run("France")
        print(f"✅ Chain response: {response}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test OpenAI LLM: {e}")
        return False

@traceable
def main():
    """Main test function"""
    print("🧪 Running LangChain tests...")
    print("=" * 50)
    
    # Test imports
    import_success = test_langchain_import()
    components_success = test_basic_components()
    chain_success = test_simple_chain()
    llm_success = test_openai_llm()
    
    print("=" * 50)
    if all([import_success, components_success, chain_success, llm_success]):
        print("🎉 All tests passed! LangChain is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())