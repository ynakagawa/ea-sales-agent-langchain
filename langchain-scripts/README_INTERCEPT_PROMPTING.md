# Intercept Prompting with LangChain

A comprehensive guide to intercept prompting - a powerful technique for intercepting and modifying prompts and responses at different stages of LLM chain execution.

## 🎯 What is Intercept Prompting?

**Intercept prompting** is a technique where you intercept and modify prompts and responses at different stages of LLM chain execution. This allows you to:

- **Modify prompts** before they're sent to the LLM
- **Modify responses** after they're received from the LLM
- **Add context** or safety instructions automatically
- **Monitor** and **debug** chain execution
- **Implement A/B testing** for different prompt versions
- **Add formatting** or validation rules

## 🔧 Key Components

### 1. Callback Handlers
The core mechanism for intercepting prompts and responses:

```python
from langchain.callbacks import BaseCallbackHandler

class InterceptCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Intercept prompts before sending to LLM
        pass
    
    def on_llm_end(self, response, **kwargs):
        # Intercept responses after receiving from LLM
        pass
```

### 2. Interception Stages
Different points where you can intercept:

- **`llm_start`** - Before prompt is sent to LLM
- **`llm_end`** - After response is received from LLM
- **`llm_error`** - When errors occur
- **`chain_start/end`** - At chain boundaries

### 3. Modification Strategies
Different approaches to modifying content:

- **Pre-processing** - Modify prompts before sending
- **Post-processing** - Modify responses after receiving
- **Conditional Logic** - Apply modifications based on content
- **Strategy Selection** - Choose different strategies

## 🚀 Use Cases

### 1. Safety Filtering
Add safety instructions automatically:

```python
def _add_safety_interception(self, prompt: str) -> str:
    safety_prefix = "Please ensure your response is safe, ethical, and follows best practices. "
    return f"{safety_prefix}{prompt}"
```

### 2. Context Enhancement
Add relevant context to prompts:

```python
def _add_context_interception(self, prompt: str) -> str:
    context_prefix = "Context: You are helping a user with a technical question. "
    return f"{context_prefix}{prompt}"
```

### 3. Formatting
Ensure consistent output format:

```python
def _add_formatting_interception(self, prompt: str) -> str:
    if "code" in prompt.lower():
        format_suffix = "\n\nPlease format any code examples clearly."
        return f"{prompt}{format_suffix}"
    return prompt
```

### 4. Monitoring
Track prompt/response patterns:

```python
def on_llm_start(self, serialized, prompts, **kwargs):
    for prompt in prompts:
        self.intercepted_prompts.append({
            "timestamp": datetime.now(),
            "prompt": prompt,
            "length": len(prompt)
        })
```

### 5. Debugging
Inspect chain execution:

```python
def on_llm_end(self, response, **kwargs):
    print(f"Response received: {response.generations[0][0].text}")
    print(f"Metadata: {kwargs}")
```

### 6. A/B Testing
Compare different prompt versions:

```python
def run_ab_test(self, prompt: str, variant_a: str, variant_b: str):
    # Test variant A
    result_a = self.run_with_interception(variant_a + prompt)
    # Test variant B
    result_b = self.run_with_interception(variant_b + prompt)
    # Compare results
    return self.compare_results(result_a, result_b)
```

## 📊 Implementation Examples

### Basic Intercept Prompting

```python
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

# Create intercept handler
intercept_handler = InterceptCallbackHandler("my_handler")

# Create chain with interception
chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ]),
    callbacks=[intercept_handler]
)

# Run with interception
result = chain.run("Explain Python functions")
```

### Advanced Strategy-Based Interception

```python
class AdvancedPromptInterceptor:
    def __init__(self):
        self.strategies = {
            "safety": self._add_safety,
            "clarity": self._add_clarity,
            "formatting": self._add_formatting
        }
    
    def run_with_strategies(self, prompt: str, strategies: List[str]):
        modified_prompt = prompt
        for strategy in strategies:
            modified_prompt = self.strategies[strategy](modified_prompt)
        return self.run_with_interception(modified_prompt)
```

### Conversation Interception

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class ConversationInterceptor:
    def __init__(self):
        self.memory = ConversationBufferMemory()
        self.intercept_handler = InterceptCallbackHandler()
    
    def chat_with_interception(self, message: str):
        chain = ConversationChain(
            llm=ChatOpenAI(),
            memory=self.memory,
            callbacks=[self.intercept_handler]
        )
        return chain.run(message)
```

## 🔍 Monitoring and Analytics

### Track Interceptions

```python
class MonitoringInterceptor(BaseCallbackHandler):
    def __init__(self):
        self.metrics = {
            "total_prompts": 0,
            "total_responses": 0,
            "prompt_modifications": 0,
            "response_modifications": 0,
            "average_prompt_length": 0,
            "average_response_length": 0
        }
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.metrics["total_prompts"] += len(prompts)
        self.metrics["average_prompt_length"] = sum(len(p) for p in prompts) / len(prompts)
    
    def on_llm_end(self, response, **kwargs):
        self.metrics["total_responses"] += 1
        response_text = response.generations[0][0].text
        self.metrics["average_response_length"] = len(response_text)
```

### Performance Monitoring

```python
import time

class PerformanceInterceptor(BaseCallbackHandler):
    def __init__(self):
        self.start_times = {}
        self.response_times = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_times[id(prompts)] = time.time()
    
    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and 'token_usage' in response.llm_output:
            token_usage = response.llm_output['token_usage']
            print(f"Tokens used: {token_usage}")
```

## 🛡️ Safety and Content Filtering

### Content Filtering

```python
class SafetyInterceptor(BaseCallbackHandler):
    def __init__(self):
        self.blocked_keywords = ["harmful", "dangerous", "illegal"]
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        for prompt in prompts:
            if any(keyword in prompt.lower() for keyword in self.blocked_keywords):
                raise ValueError("Prompt contains blocked keywords")
    
    def on_llm_end(self, response, **kwargs):
        response_text = response.generations[0][0].text
        if any(keyword in response_text.lower() for keyword in self.blocked_keywords):
            # Modify response to remove harmful content
            modified_response = self._filter_content(response_text)
            response.generations[0][0].text = modified_response
```

### Bias Detection

```python
class BiasInterceptor(BaseCallbackHandler):
    def __init__(self):
        self.bias_indicators = ["always", "never", "everyone", "nobody"]
    
    def on_llm_end(self, response, **kwargs):
        response_text = response.generations[0][0].text
        bias_score = self._calculate_bias_score(response_text)
        if bias_score > 0.7:
            print(f"⚠️ High bias detected: {bias_score}")
```

## 🔧 Advanced Patterns

### Conditional Interception

```python
class ConditionalInterceptor(BaseCallbackHandler):
    def __init__(self, conditions: Dict[str, callable]):
        self.conditions = conditions
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        for prompt in prompts:
            for condition_name, condition_func in self.conditions.items():
                if condition_func(prompt):
                    print(f"Condition '{condition_name}' met for prompt")
                    # Apply specific logic
```

### Multi-Stage Interception

```python
class MultiStageInterceptor(BaseCallbackHandler):
    def __init__(self):
        self.stages = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.stages.append("prompt_intercepted")
        # Stage 1: Content analysis
        # Stage 2: Safety check
        # Stage 3: Enhancement
        # Stage 4: Final formatting
    
    def on_llm_end(self, response, **kwargs):
        self.stages.append("response_intercepted")
        # Stage 1: Response validation
        # Stage 2: Content filtering
        # Stage 3: Formatting
        # Stage 4: Quality check
```

## 📈 Best Practices

### 1. Performance Considerations
- Keep interception logic lightweight
- Use caching for expensive operations
- Monitor performance impact
- Implement timeouts for long-running interceptors

### 2. Error Handling
- Always handle exceptions in interceptors
- Provide fallback behavior
- Log errors for debugging
- Don't let interceptors break the main flow

### 3. Testing
- Test interceptors in isolation
- Use mock LLM responses
- Validate modification logic
- Test edge cases and error conditions

### 4. Monitoring
- Track interception metrics
- Monitor performance impact
- Log important events
- Set up alerts for issues

## 🚀 Running the Examples

### Basic Demo
```bash
python intercept_prompting_langchain.py
```

### Environment Setup
```bash
# Install dependencies
pip install langchain langchain-community openai python-dotenv

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"
```

### Custom Implementation
```python
# Create your own interceptor
class MyCustomInterceptor(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        # Your custom logic here
        pass
    
    def on_llm_end(self, response, **kwargs):
        # Your custom logic here
        pass

# Use with your chain
chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=your_prompt,
    callbacks=[MyCustomInterceptor()]
)
```

## 🔗 Related Concepts

- **Prompt Engineering** - Designing effective prompts
- **Chain of Thought** - Step-by-step reasoning
- **Few-Shot Learning** - Learning from examples
- **Retrieval-Augmented Generation** - Using external knowledge
- **Constitutional AI** - Safety and alignment

## 📚 Additional Resources

- [LangChain Callbacks Documentation](https://python.langchain.com/docs/modules/callbacks/)
- [LangChain Chains Documentation](https://python.langchain.com/docs/modules/chains/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 