#!/usr/bin/env python3
"""
Intercept Prompting Demo - Core Concepts
This script demonstrates intercept prompting concepts without requiring an API key.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

class MockLLM:
    """Mock LLM for demonstration purposes"""
    
    def __init__(self, name: str = "mock-llm"):
        self.name = name
    
    def generate(self, prompts: List[str], **kwargs):
        """Mock response generation"""
        responses = []
        for prompt in prompts:
            # Simulate different responses based on prompt content
            if "python" in prompt.lower():
                response = "Here's how to write Python code: def example(): return 'Hello World'"
            elif "function" in prompt.lower():
                response = "A function is a reusable block of code that performs a specific task."
            elif "safety" in prompt.lower():
                response = "Always ensure your code follows security best practices and handles errors gracefully."
            else:
                response = "I can help you with that. Here's a general explanation..."
            
            responses.append(response)
        
        return MockResponse(responses)

class MockResponse:
    """Mock response object"""
    
    def __init__(self, responses: List[str]):
        self.generations = [[MockGeneration(response)] for response in responses]

class MockGeneration:
    """Mock generation object"""
    
    def __init__(self, text: str):
        self.text = text

class InterceptCallbackHandler:
    """Custom callback handler to intercept prompts and responses"""
    
    def __init__(self, name: str = "intercept_handler"):
        self.name = name
        self.intercepted_prompts = []
        self.intercepted_responses = []
        self.interception_log = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Intercept prompts before they're sent to the LLM"""
        print(f"🔍 [{self.name}] Intercepting {len(prompts)} prompt(s)")
        
        for i, prompt in enumerate(prompts):
            intercepted_data = {
                "stage": "llm_start",
                "prompt_index": i,
                "original_prompt": prompt,
                "modified_prompt": self._modify_prompt(prompt),
                "timestamp": datetime.now().isoformat(),
                "metadata": kwargs
            }
            self.intercepted_prompts.append(intercepted_data)
            self.interception_log.append(f"📝 Intercepted prompt {i}: {prompt[:50]}...")
            
            print(f"📝 [{self.name}] Original prompt {i}: {prompt[:50]}...")
            print(f"🔧 [{self.name}] Modified prompt {i}: {intercepted_data['modified_prompt'][:50]}...")
    
    def on_llm_end(self, response, **kwargs):
        """Intercept responses after they're received from the LLM"""
        print(f"📤 [{self.name}] Intercepting response")
        
        response_text = response.generations[0][0].text if response.generations else ""
        
        intercepted_data = {
            "stage": "llm_end",
            "original_response": response_text,
            "modified_response": self._modify_response(response_text),
            "timestamp": datetime.now().isoformat(),
            "metadata": kwargs
        }
        self.intercepted_responses.append(intercepted_data)
        self.interception_log.append(f"📤 Intercepted response: {response_text[:50]}...")
        
        print(f"📤 [{self.name}] Original response: {response_text[:50]}...")
        print(f"🔧 [{self.name}] Modified response: {intercepted_data['modified_response'][:50]}...")
    
    def _modify_prompt(self, prompt: str) -> str:
        """Modify the prompt before sending to LLM"""
        modified_prompt = prompt
        
        # Add safety context if not present
        if "safety" not in prompt.lower() and "harmful" not in prompt.lower():
            modified_prompt = f"{prompt}\n\nPlease ensure your response is helpful, accurate, and safe."
        
        # Add clarity instructions for complex queries
        if len(prompt.split()) > 20:
            modified_prompt = f"{modified_prompt}\n\nPlease provide a clear, structured response."
        
        # Add context for technical questions
        if any(word in prompt.lower() for word in ["code", "programming", "function", "class"]):
            modified_prompt = f"{modified_prompt}\n\nPlease include practical examples in your response."
        
        return modified_prompt
    
    def _modify_response(self, response: str) -> str:
        """Modify the response after receiving from LLM"""
        modified_response = response
        
        # Add disclaimer for technical content
        if any(word in response.lower() for word in ["code", "script", "command", "api", "def", "class"]):
            modified_response = f"{response}\n\n⚠️ Note: Please review and test any code before using in production."
        
        # Add confidence indicator
        if "i don't know" in response.lower() or "i'm not sure" in response.lower():
            modified_response = f"{response}\n\n🤔 This response has low confidence. Please verify the information."
        
        # Add helpful links for technical topics
        if "python" in response.lower():
            modified_response = f"{response}\n\n🔗 Helpful resources: Python Documentation (https://docs.python.org/)"
        
        return modified_response

class PromptInterceptor:
    """Main class for intercept prompting functionality"""
    
    def __init__(self):
        self.llm = MockLLM("demo-llm")
        self.intercept_handler = InterceptCallbackHandler("demo_interceptor")
    
    def run_with_interception(self, prompt: str) -> Dict[str, Any]:
        """Run a prompt with full interception capabilities"""
        
        print(f"\n🚀 Running prompt with interception: {prompt[:50]}...")
        
        # Simulate the interception process
        self.intercept_handler.on_llm_start(
            {"name": "mock-llm"}, 
            [prompt], 
            temperature=0.7
        )
        
        # Generate response
        response = self.llm.generate([prompt])
        
        # Intercept response
        self.intercept_handler.on_llm_end(response, model="mock-llm")
        
        # Return comprehensive results
        return {
            "final_result": response.generations[0][0].text,
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
    
    def __init__(self):
        super().__init__()
        self.interception_strategies = {
            "safety": self._add_safety_interception,
            "clarity": self._add_clarity_interception,
            "context": self._add_context_interception,
            "formatting": self._add_formatting_interception,
            "technical": self._add_technical_interception
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
    
    def _add_technical_interception(self, prompt: str) -> str:
        """Add technical-related interceptions"""
        if any(word in prompt.lower() for word in ["python", "javascript", "java", "c++"]):
            tech_suffix = "\n\nPlease include practical examples and best practices for this technology."
            return f"{prompt}{tech_suffix}"
        return prompt
    
    def run_with_strategies(self, prompt: str, strategies: List[str] = None) -> Dict[str, Any]:
        """Run with specific interception strategies"""
        
        if strategies is None:
            strategies = ["safety", "clarity"]
        
        print(f"\n🎯 Applying strategies: {', '.join(strategies)}")
        
        # Apply strategies to the prompt
        modified_prompt = prompt
        for strategy in strategies:
            if strategy in self.interception_strategies:
                original_prompt = modified_prompt
                modified_prompt = self.interception_strategies[strategy](modified_prompt)
                if original_prompt != modified_prompt:
                    print(f"  ✅ Applied {strategy} strategy")
        
        return self.run_with_interception(modified_prompt)

def demo_basic_interception():
    """Demonstrate basic intercept prompting"""
    print("🚀 Basic Intercept Prompting Demo")
    print("=" * 50)
    
    interceptor = PromptInterceptor()
    
    # Test queries
    test_prompts = [
        "Explain how to write a Python function",
        "What is machine learning?",
        "How do I create a web application?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i} ---")
        result = interceptor.run_with_interception(prompt)
        
        print(f"\n✅ Final Result: {result['final_result'][:100]}...")
        
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
    
    interceptor = AdvancedPromptInterceptor()
    
    # Test with different strategy combinations
    test_cases = [
        {
            "prompt": "Write a Python function to calculate fibonacci numbers",
            "strategies": ["safety", "clarity", "formatting", "technical"]
        },
        {
            "prompt": "Explain the concept of recursion",
            "strategies": ["clarity", "context"]
        },
        {
            "prompt": "How do I implement authentication in a web app?",
            "strategies": ["safety", "technical", "formatting"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Advanced Test {i} ---")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Strategies: {', '.join(test_case['strategies'])}")
        
        result = interceptor.run_with_strategies(
            test_case['prompt'], 
            test_case['strategies']
        )
        
        print(f"\n✅ Final Result: {result['final_result'][:100]}...")
        
        # Show what was modified
        summary = result['interception_summary']
        modified_prompts = sum(1 for p in summary['prompt_modifications'] if p['was_modified'])
        modified_responses = sum(1 for r in summary['response_modifications'] if r['was_modified'])
        
        print(f"📊 Modifications: {modified_prompts} prompts, {modified_responses} responses")

def demo_interception_patterns():
    """Demonstrate different interception patterns"""
    print("\n🎯 Interception Patterns Demo")
    print("=" * 50)
    
    patterns = {
        "Pre-processing": "Modify prompts before sending to LLM",
        "Post-processing": "Modify responses after receiving from LLM",
        "Conditional Logic": "Apply modifications based on content analysis",
        "Strategy Selection": "Choose different strategies based on requirements",
        "Monitoring": "Track all modifications and patterns"
    }
    
    for pattern_name, description in patterns.items():
        print(f"\n🔧 {pattern_name}:")
        print(f"   {description}")
        
        # Show example implementation
        if pattern_name == "Pre-processing":
            print("   Example: Add safety instructions to all prompts")
        elif pattern_name == "Post-processing":
            print("   Example: Add disclaimers to technical responses")
        elif pattern_name == "Conditional Logic":
            print("   Example: Add code formatting only for programming questions")
        elif pattern_name == "Strategy Selection":
            print("   Example: Choose safety+clarity for beginners, technical for experts")
        elif pattern_name == "Monitoring":
            print("   Example: Track modification frequency and effectiveness")

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
    print("🚀 LangChain Intercept Prompting Demo")
    print("=" * 60)
    
    show_intercept_prompting_concepts()
    demo_basic_interception()
    demo_advanced_interception()
    demo_interception_patterns()
    
    print("\n✨ Demo completed!")
    print("\n💡 **Key Takeaways:**")
    print("• Intercept prompting allows you to modify prompts and responses")
    print("• You can add safety, clarity, and formatting automatically")
    print("• Different strategies can be combined for powerful effects")
    print("• Monitoring and debugging become much easier")
    print("• This technique is essential for production LLM applications")

if __name__ == "__main__":
    main() 