#!/usr/bin/env python3
"""
Intercept Prompting with LangChain
This script demonstrates how to intercept and modify prompts at different stages of LLM chain execution.
"""

import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterceptCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to intercept prompts and responses"""
    
    def __init__(self, name: str = "intercept_handler"):
        self.name = name
        self.intercepted_prompts = []
        self.intercepted_responses = []
        self.intercepted_metadata = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Intercept prompts before they're sent to the LLM"""
        logger.info(f"🔍 [{self.name}] Intercepting {len(prompts)} prompt(s)")
        
        for i, prompt in enumerate(prompts):
            intercepted_data = {
                "stage": "llm_start",
                "prompt_index": i,
                "original_prompt": prompt,
                "modified_prompt": self._modify_prompt(prompt),
                "metadata": kwargs
            }
            self.intercepted_prompts.append(intercepted_data)
            logger.info(f"📝 [{self.name}] Original prompt {i}: {prompt[:100]}...")
            logger.info(f"🔧 [{self.name}] Modified prompt {i}: {intercepted_data['modified_prompt'][:100]}...")
    
    def on_llm_end(self, response, **kwargs):
        """Intercept responses after they're received from the LLM"""
        logger.info(f"📤 [{self.name}] Intercepting response")
        
        intercepted_data = {
            "stage": "llm_end",
            "original_response": response.generations[0][0].text if response.generations else "",
            "modified_response": self._modify_response(response.generations[0][0].text if response.generations else ""),
            "metadata": kwargs
        }
        self.intercepted_responses.append(intercepted_data)
        logger.info(f"📤 [{self.name}] Original response: {intercepted_data['original_response'][:100]}...")
        logger.info(f"🔧 [{self.name}] Modified response: {intercepted_data['modified_response'][:100]}...")
    
    def _modify_prompt(self, prompt: str) -> str:
        """Modify the prompt before sending to LLM"""
        # Example modifications:
        # 1. Add context
        # 2. Modify tone
        # 3. Add safety instructions
        # 4. Enhance clarity
        
        modified_prompt = prompt
        
        # Add safety context if not present
        if "safety" not in prompt.lower() and "harmful" not in prompt.lower():
            modified_prompt = f"{prompt}\n\nPlease ensure your response is helpful, accurate, and safe."
        
        # Add clarity instructions for complex queries
        if len(prompt.split()) > 50:
            modified_prompt = f"{prompt}\n\nPlease provide a clear, structured response."
        
        return modified_prompt
    
    def _modify_response(self, response: str) -> str:
        """Modify the response after receiving from LLM"""
        # Example modifications:
        # 1. Add disclaimers
        # 2. Format output
        # 3. Add metadata
        # 4. Filter content
        
        modified_response = response
        
        # Add disclaimer for technical content
        if any(word in response.lower() for word in ["code", "script", "command", "api"]):
            modified_response = f"{response}\n\n⚠️ Note: Please review and test any code before using in production."
        
        # Add confidence indicator
        if "i don't know" in response.lower() or "i'm not sure" in response.lower():
            modified_response = f"{response}\n\n🤔 This response has low confidence. Please verify the information."
        
        return modified_response

class PromptInterceptor:
    """Main class for intercept prompting functionality"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        self.intercept_handler = InterceptCallbackHandler("main_interceptor")
    
    def create_intercept_chain(self, prompt_template: str, system_message: str = "") -> LLMChain:
        """Create a chain with intercept capabilities"""
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message or "You are a helpful AI assistant."),
            ("human", prompt_template)
        ])
        
        # Create chain with intercept handler
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            callbacks=[self.intercept_handler]
        )
        
        return chain
    
    def run_with_interception(self, prompt_template: str, inputs: Dict[str, Any], 
                            system_message: str = "") -> Dict[str, Any]:
        """Run a chain with full interception capabilities"""
        
        chain = self.create_intercept_chain(prompt_template, system_message)
        
        # Run the chain
        result = chain.run(inputs)
        
        # Return comprehensive results
        return {
            "final_result": result,
            "intercepted_prompts": self.intercept_handler.intercepted_prompts,
            "intercepted_responses": self.intercept_handler.intercepted_responses,
            "interception_summary": self._create_interception_summary()
        }
    
    def _create_interception_summary(self) -> Dict[str, Any]:
        """Create a summary of all interceptions"""
        return {
            "total_prompts_intercepted": len(self.intercept_handler.intercepted_prompts),
            "total_responses_intercepted": len(self.intercept_handler.intercepted_responses),
            "prompt_modifications": [
                {
                    "index": i,
                    "original_length": len(data["original_prompt"]),
                    "modified_length": len(data["modified_prompt"]),
                    "was_modified": data["original_prompt"] != data["modified_prompt"]
                }
                for i, data in enumerate(self.intercept_handler.intercepted_prompts)
            ],
            "response_modifications": [
                {
                    "index": i,
                    "original_length": len(data["original_response"]),
                    "modified_length": len(data["modified_response"]),
                    "was_modified": data["original_response"] != data["modified_response"]
                }
                for i, data in enumerate(self.intercept_handler.intercepted_responses)
            ]
        }

class AdvancedPromptInterceptor(PromptInterceptor):
    """Advanced intercept prompting with multiple strategies"""
    
    def __init__(self, openai_api_key: str):
        super().__init__(openai_api_key)
        self.interception_strategies = {
            "safety": self._add_safety_interception,
            "clarity": self._add_clarity_interception,
            "context": self._add_context_interception,
            "formatting": self._add_formatting_interception
        }
    
    def _add_safety_interception(self, prompt: str) -> str:
        """Add safety-related interceptions"""
        safety_prefix = "Please ensure your response is safe, ethical, and follows best practices. "
        return f"{safety_prefix}{prompt}"
    
    def _add_clarity_interception(self, prompt: str) -> str:
        """Add clarity-related interceptions"""
        if "explain" in prompt.lower() or "how" in prompt.lower():
            clarity_suffix = "\n\nPlease provide a clear, step-by-step explanation."
            return f"{prompt}{clarity_suffix}"
        return prompt
    
    def _add_context_interception(self, prompt: str) -> str:
        """Add context-related interceptions"""
        context_prefix = "Context: You are helping a user with a technical question. "
        return f"{context_prefix}{prompt}"
    
    def _add_formatting_interception(self, prompt: str) -> str:
        """Add formatting-related interceptions"""
        if "code" in prompt.lower() or "programming" in prompt.lower():
            format_suffix = "\n\nPlease format any code examples clearly with proper syntax highlighting."
            return f"{prompt}{format_suffix}"
        return prompt
    
    def run_with_strategies(self, prompt_template: str, inputs: Dict[str, Any], 
                          strategies: List[str] = None, system_message: str = "") -> Dict[str, Any]:
        """Run with specific interception strategies"""
        
        if strategies is None:
            strategies = ["safety", "clarity"]
        
        # Apply strategies to the prompt template
        modified_template = prompt_template
        for strategy in strategies:
            if strategy in self.interception_strategies:
                modified_template = self.interception_strategies[strategy](modified_template)
        
        return self.run_with_interception(modified_template, inputs, system_message)

class ConversationInterceptor:
    """Intercept prompting for conversation chains"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        self.memory = ConversationBufferMemory()
        self.intercept_handler = InterceptCallbackHandler("conversation_interceptor")
    
    def create_conversation_chain(self) -> ConversationChain:
        """Create a conversation chain with interception"""
        return ConversationChain(
            llm=self.llm,
            memory=self.memory,
            callbacks=[self.intercept_handler]
        )
    
    def chat_with_interception(self, message: str) -> Dict[str, Any]:
        """Chat with full interception capabilities"""
        chain = self.create_conversation_chain()
        
        # Run the conversation
        result = chain.run(message)
        
        return {
            "response": result,
            "intercepted_prompts": self.intercept_handler.intercepted_prompts,
            "intercepted_responses": self.intercept_handler.intercepted_responses,
            "conversation_history": self.memory.chat_memory.messages
        }

def demo_basic_interception():
    """Demonstrate basic intercept prompting"""
    print("🚀 Basic Intercept Prompting Demo")
    print("=" * 50)
    
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found. Running in demo mode.")
        return
    
    interceptor = PromptInterceptor(openai_api_key)
    
    # Example 1: Simple prompt interception
    prompt_template = "Explain how to write a Python function"
    inputs = {}
    
    print("📝 Running with prompt interception...")
    result = interceptor.run_with_interception(prompt_template, inputs)
    
    print(f"✅ Final Result: {result['final_result'][:200]}...")
    print(f"📊 Intercepted {len(result['intercepted_prompts'])} prompts")
    print(f"📊 Intercepted {len(result['intercepted_responses'])} responses")
    
    # Show interception summary
    summary = result['interception_summary']
    print(f"\n📈 Interception Summary:")
    print(f"  • Prompts intercepted: {summary['total_prompts_intercepted']}")
    print(f"  • Responses intercepted: {summary['total_responses_intercepted']}")
    print(f"  • Prompts modified: {sum(1 for p in summary['prompt_modifications'] if p['was_modified'])}")
    print(f"  • Responses modified: {sum(1 for r in summary['response_modifications'] if r['was_modified'])}")

def demo_advanced_interception():
    """Demonstrate advanced intercept prompting with strategies"""
    print("\n🔧 Advanced Intercept Prompting Demo")
    print("=" * 50)
    
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found. Running in demo mode.")
        return
    
    interceptor = AdvancedPromptInterceptor(openai_api_key)
    
    # Example with multiple strategies
    prompt_template = "Write a Python function to calculate fibonacci numbers"
    inputs = {}
    strategies = ["safety", "clarity", "formatting"]
    
    print("📝 Running with advanced interception strategies...")
    result = interceptor.run_with_strategies(prompt_template, inputs, strategies)
    
    print(f"✅ Final Result: {result['final_result'][:200]}...")
    
    # Show what strategies were applied
    print(f"\n🎯 Applied Strategies: {', '.join(strategies)}")
    print(f"📊 Total interceptions: {len(result['intercepted_prompts'])}")

def demo_conversation_interception():
    """Demonstrate conversation interception"""
    print("\n💬 Conversation Intercept Prompting Demo")
    print("=" * 50)
    
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found. Running in demo mode.")
        return
    
    interceptor = ConversationInterceptor(openai_api_key)
    
    # Simulate a conversation
    messages = [
        "Hello! Can you help me with Python programming?",
        "How do I write a function that sorts a list?",
        "What about error handling in Python?"
    ]
    
    print("💬 Running conversation with interception...")
    
    for i, message in enumerate(messages, 1):
        print(f"\n--- Message {i} ---")
        print(f"User: {message}")
        
        result = interceptor.chat_with_interception(message)
        
        print(f"AI: {result['response'][:100]}...")
        print(f"📊 Intercepted {len(result['intercepted_prompts'])} prompts")
        print(f"📊 Intercepted {len(result['intercepted_responses'])} responses")

def show_intercept_prompting_concepts():
    """Show the key concepts of intercept prompting"""
    print("\n📚 Intercept Prompting Concepts")
    print("=" * 50)
    
    print("🎯 **What is Intercept Prompting?**")
    print("Intercept prompting is a technique where you intercept and modify")
    print("prompts and responses at different stages of LLM chain execution.")
    print()
    
    print("🔧 **Key Components:**")
    print("1. **Callback Handlers** - Intercept at specific stages")
    print("2. **Prompt Modification** - Modify prompts before LLM")
    print("3. **Response Modification** - Modify responses after LLM")
    print("4. **Strategy Patterns** - Apply different interception strategies")
    print()
    
    print("📊 **Interception Stages:**")
    print("• **llm_start** - Before prompt is sent to LLM")
    print("• **llm_end** - After response is received from LLM")
    print("• **llm_error** - When errors occur")
    print("• **chain_start/end** - At chain boundaries")
    print()
    
    print("🎯 **Use Cases:**")
    print("• **Safety Filtering** - Add safety instructions")
    print("• **Context Enhancement** - Add relevant context")
    print("• **Formatting** - Ensure consistent output format")
    print("• **Monitoring** - Track prompt/response patterns")
    print("• **Debugging** - Inspect chain execution")
    print("• **A/B Testing** - Compare different prompt versions")
    print()
    
    print("🔧 **Implementation Patterns:**")
    print("1. **Pre-processing** - Modify prompts before sending")
    print("2. **Post-processing** - Modify responses after receiving")
    print("3. **Conditional Logic** - Apply modifications based on content")
    print("4. **Strategy Selection** - Choose different strategies")
    print("5. **Monitoring** - Track all modifications")

def main():
    """Main function to run all demos"""
    print("🚀 LangChain Intercept Prompting Examples")
    print("=" * 60)
    
    show_intercept_prompting_concepts()
    
    # Run demos if API key is available
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key:
        demo_basic_interception()
        demo_advanced_interception()
        demo_conversation_interception()
    else:
        print("\n❌ Demo Mode - OpenAI API key not available")
        print("To run full demos, set OPENAI_API_KEY in your .env file")
    
    print("\n✨ Demo completed!")

if __name__ == "__main__":
    main() 