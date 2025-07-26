#!/usr/bin/env python3
"""
LangChain with Azure OpenAI Test Script
Comprehensive test of LangChain features with Azure OpenAI Service
"""

import os
import warnings
from typing import List, Dict, Any
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

class LangChainAzureOpenAITest:
    """Test class for LangChain with Azure OpenAI integration"""
    
    def __init__(self):
        """Initialize LangChain Azure OpenAI test components"""
        # Get Azure OpenAI credentials
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.chat_deployment = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'gpt-35-turbo')
        self.embedding_deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        
        # Check if credentials are available
        if not self.azure_endpoint or not self.azure_api_key:
            print("⚠️  Azure OpenAI credentials not found!")
            print("Please set the following environment variables:")
            print("  - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint")
            print("  - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key")
            print("  - AZURE_OPENAI_CHAT_DEPLOYMENT: Chat model deployment (optional)")
            print("  - AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Embedding model deployment (optional)")
            print("  - AZURE_OPENAI_API_VERSION: API version (optional)")
            return
        
        # Initialize components
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LangChain components with Azure OpenAI"""
        try:
            # Initialize Azure OpenAI Chat Model
            self.chat_model = AzureChatOpenAI(
                azure_deployment=self.chat_deployment,
                openai_api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                temperature=0.1
            )
            print(f"✅ Initialized Azure OpenAI Chat Model with deployment: {self.chat_deployment}")
            
            # Initialize Azure OpenAI Embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=self.embedding_deployment,
                openai_api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key
            )
            print(f"✅ Initialized Azure OpenAI Embeddings with deployment: {self.embedding_deployment}")
            
        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
    
    def test_basic_chains(self):
        """Test basic LangChain chains with Azure OpenAI"""
        print("\n⛓️ Testing Basic LangChain Chains")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Test LLMChain
            template = """
            You are a helpful AI assistant. Please provide a brief explanation of {topic}.
            Keep your response under 100 words and focus on key points.
            """
            
            prompt = PromptTemplate(
                input_variables=["topic"],
                template=template
            )
            
            chain = LLMChain(llm=self.chat_model, prompt=prompt)
            
            topics = ["machine learning", "cloud computing", "artificial intelligence"]
            for topic in topics:
                response = chain.run(topic=topic)
                print(f"\n📝 {topic}: {response}")
            
        except Exception as e:
            print(f"❌ Basic chains test failed: {e}")
    
    def test_conversation_memory(self):
        """Test conversation memory with Azure OpenAI"""
        print("\n🧠 Testing Conversation Memory")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Test ConversationBufferMemory
            memory = ConversationBufferMemory()
            conversation = ConversationChain(
                llm=self.chat_model,
                memory=memory,
                verbose=False
            )
            
            # Test conversation flow
            responses = []
            questions = [
                "What is machine learning?",
                "How does it relate to artificial intelligence?",
                "What are some common applications?",
                "Can you give me a specific example?",
                "What is the biggest risk for Adobe?"
            ]
            
            for question in questions:
                response = conversation.predict(input=question)
                responses.append(response)
                print(f"\n🤖 Q: {question}")
                print(f"💬 A: {response[:150]}...")
            
            # Show memory contents
            print(f"\n📚 Memory Buffer: {len(memory.buffer)} characters")
            print(f"💾 Memory Variables: {memory.memory_variables}")
            
        except Exception as e:
            print(f"❌ Conversation memory test failed: {e}")
    
    def test_sequential_chains(self):
        """Test sequential chains with Azure OpenAI"""
        print("\n🔄 Testing Sequential Chains")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Create first chain - topic analysis
            topic_template = """
            Analyze the following topic and provide key insights:
            Topic: {topic}
            
            Provide:
            1. Definition
            2. Key concepts
            3. Applications
            """
            
            topic_prompt = PromptTemplate(
                input_variables=["topic"],
                template=topic_template
            )
            
            topic_chain = LLMChain(llm=self.chat_model, prompt=topic_prompt)
            
            # Create second chain - summary
            summary_template = """
            Create a concise summary of the following analysis:
            {analysis}
            
            Summary should be:
            - Under 50 words
            - Focus on main points
            - Easy to understand
            """
            
            summary_prompt = PromptTemplate(
                input_variables=["analysis"],
                template=summary_template
            )
            
            summary_chain = LLMChain(llm=self.chat_model, prompt=summary_prompt)
            
            # Create sequential chain
            overall_chain = SimpleSequentialChain(
                chains=[topic_chain, summary_chain],
                verbose=True
            )
            
            # Test the chain
            topic = "blockchain technology"
            result = overall_chain.run(topic)
            print(f"\n📊 Sequential Chain Result:")
            print(f"Topic: {topic}")
            print(f"Summary: {result}")
            
        except Exception as e:
            print(f"❌ Sequential chains test failed: {e}")
    
    def test_output_parsers(self):
        """Test output parsers with Azure OpenAI"""
        print("\n🔧 Testing Output Parsers")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Define Pydantic model for structured output
            class TechnologyAnalysis(BaseModel):
                name: str = Field(description="Technology name")
                category: str = Field(description="Technology category")
                key_features: List[str] = Field(description="Key features")
                applications: List[str] = Field(description="Common applications")
                complexity: str = Field(description="Complexity level (Low/Medium/High)")
            
            # Create parser
            parser = PydanticOutputParser(pydantic_object=TechnologyAnalysis)
            
            # Create prompt template
            template = """
            Analyze the following technology and provide structured information:
            Technology: {technology}
            
            {format_instructions}
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["technology"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            # Create chain
            chain = LLMChain(llm=self.chat_model, prompt=prompt)
            
            # Test with different technologies
            technologies = ["artificial intelligence", "cloud computing", "blockchain"]
            
            for tech in technologies:
                try:
                    response = chain.run(technology=tech)
                    parsed_result = parser.parse(response)
                    print(f"\n🔍 {tech.upper()}:")
                    print(f"   Category: {parsed_result.category}")
                    print(f"   Features: {', '.join(parsed_result.key_features[:3])}")
                    print(f"   Applications: {', '.join(parsed_result.applications[:3])}")
                    print(f"   Complexity: {parsed_result.complexity}")
                except Exception as parse_error:
                    print(f"❌ Failed to parse {tech}: {parse_error}")
            
        except Exception as e:
            print(f"❌ Output parsers test failed: {e}")
    
    def test_vector_store_qa(self):
        """Test vector store Q&A with Azure OpenAI"""
        print("\n🔍 Testing Vector Store Q&A")
        print("=" * 50)
        
        if not self.embeddings or not self.chat_model:
            print("❌ Embeddings or chat model not initialized")
            return
        
        try:
            # Create sample documents
            documents = [
                Document(
                    page_content="Azure OpenAI Service provides enterprise-grade access to OpenAI's powerful language models with built-in security and compliance features.",
                    metadata={"source": "azure_docs", "topic": "overview"}
                ),
                Document(
                    page_content="The service supports various models including GPT-3.5, GPT-4, and embedding models, all deployed securely in your Azure environment.",
                    metadata={"source": "azure_docs", "topic": "models"}
                ),
                Document(
                    page_content="Enterprise customers can use Azure OpenAI Service for applications like chatbots, content generation, and data analysis with full control over their data.",
                    metadata={"source": "azure_docs", "topic": "use_cases"}
                ),
                Document(
                    page_content="LangChain is a framework for developing applications powered by language models, providing tools for building chains, agents, and memory systems.",
                    metadata={"source": "langchain_docs", "topic": "overview"}
                ),
                Document(
                    page_content="LangChain supports various integrations including OpenAI, Azure OpenAI, and other LLM providers, making it easy to build AI applications.",
                    metadata={"source": "langchain_docs", "topic": "integrations"}
                )
            ]
            
            print(f"✅ Created {len(documents)} documents")
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            print(f"✅ Split into {len(split_docs)} chunks")
            
            # Create vector store
            texts = [doc.page_content for doc in split_docs]
            metadatas = [doc.metadata for doc in split_docs]
            
            vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            print(f"✅ Created FAISS vector store")
            
            # Create retrieval QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.chat_model,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
            )
            
            # Test questions
            questions = [
                "What is Azure OpenAI Service?",
                "What models does it support?",
                "What is LangChain?",
                "How does LangChain integrate with Azure OpenAI?"
            ]
            
            for question in questions:
                try:
                    response = qa_chain.run(question)
                    print(f"\n❓ Q: {question}")
                    print(f"💡 A: {response[:200]}...")
                except Exception as qa_error:
                    print(f"❌ Failed to answer '{question}': {qa_error}")
            
        except Exception as e:
            print(f"❌ Vector store Q&A test failed: {e}")
    
    def test_agents(self):
        """Test LangChain agents with Azure OpenAI"""
        print("\n🤖 Testing LangChain Agents")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Create tools
            search = DuckDuckGoSearchRun()
            
            tools = [
                Tool(
                    name="Search",
                    func=search.run,
                    description="Useful for searching the internet for current information"
                )
            ]
            
            # Initialize agent
            agent = initialize_agent(
                tools,
                self.chat_model,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True
            )
            
            # Test agent with a question
            question = "What are the latest developments in artificial intelligence?"
            
            try:
                response = agent.run(question)
                print(f"\n🤖 Agent Response:")
                print(f"Q: {question}")
                print(f"A: {response[:300]}...")
            except Exception as agent_error:
                print(f"❌ Agent execution failed: {agent_error}")
                print("This might be due to tool limitations or network issues")
            
        except Exception as e:
            print(f"❌ Agents test failed: {e}")
    
    def test_chat_prompts(self):
        """Test chat prompt templates with Azure OpenAI"""
        print("\n💬 Testing Chat Prompt Templates")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Create chat prompt template
            template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert in {domain} with years of experience."),
                ("human", "Explain {concept} in simple terms that a beginner can understand."),
                ("assistant", "I'll explain {concept} in simple terms."),
                ("human", "Now provide a practical example of {concept}.")
            ])
            
            # Create chain
            chain = LLMChain(llm=self.chat_model, prompt=template)
            
            # Test with different domains and concepts
            test_cases = [
                ("machine learning", "neural networks"),
                ("cloud computing", "serverless architecture"),
                ("data science", "data preprocessing")
            ]
            
            for domain, concept in test_cases:
                try:
                    response = chain.run(domain=domain, concept=concept)
                    print(f"\n📚 {domain} - {concept}:")
                    print(f"   {response[:200]}...")
                except Exception as chain_error:
                    print(f"❌ Failed for {domain}/{concept}: {chain_error}")
            
        except Exception as e:
            print(f"❌ Chat prompts test failed: {e}")
    
    def test_memory_variations(self):
        """Test different types of memory with Azure OpenAI"""
        print("\n🧠 Testing Memory Variations")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Test ConversationSummaryMemory
            summary_memory = ConversationSummaryMemory(llm=self.chat_model)
            summary_conversation = ConversationChain(
                llm=self.chat_model,
                memory=summary_memory,
                verbose=False
            )
            
            # Test conversation with summary memory
            summary_questions = [
                "What is machine learning?",
                "How does it work?",
                "What are the main types?",
                "Give me an example of supervised learning."
            ]
            
            print("📝 Testing ConversationSummaryMemory:")
            for question in summary_questions:
                response = summary_conversation.predict(input=question)
                print(f"   Q: {question}")
                print(f"   A: {response[:100]}...")
            
            # Show memory summary
            print(f"\n📚 Memory Summary: {summary_memory.moving_summary_buffer}")
            
        except Exception as e:
            print(f"❌ Memory variations test failed: {e}")
    
    def test_advanced_chains(self):
        """Test advanced chain patterns with Azure OpenAI"""
        print("\n🚀 Testing Advanced Chain Patterns")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Create a multi-step analysis chain
            step1_template = """
            Analyze the following technology and identify its key components:
            Technology: {technology}
            
            Provide a structured analysis with:
            1. Core components
            2. Key features
            3. Technical requirements
            """
            
            step1_prompt = PromptTemplate(
                input_variables=["technology"],
                template=step1_template
            )
            
            step1_chain = LLMChain(llm=self.chat_model, prompt=step1_prompt)
            
            step2_template = """
            Based on this analysis, provide implementation recommendations:
            Analysis: {analysis}
            
            Provide:
            1. Implementation steps
            2. Best practices
            3. Common pitfalls to avoid
            """
            
            step2_prompt = PromptTemplate(
                input_variables=["analysis"],
                template=step2_template
            )
            
            step2_chain = LLMChain(llm=self.chat_model, prompt=step2_prompt)
            
            # Test the advanced chain
            technology = "microservices architecture"
            
            print(f"🔍 Analyzing: {technology}")
            
            # Step 1: Analysis
            analysis = step1_chain.run(technology=technology)
            print(f"\n📊 Analysis: {analysis[:200]}...")
            
            # Step 2: Recommendations
            recommendations = step2_chain.run(analysis=analysis)
            print(f"\n💡 Recommendations: {recommendations[:200]}...")
            
        except Exception as e:
            print(f"❌ Advanced chains test failed: {e}")
    
    def run_all_tests(self):
        """Run all LangChain Azure OpenAI tests"""
        print("🚀 LangChain with Azure OpenAI Integration Test")
        print("=" * 60)
        
        # Check configuration
        print("🔍 Configuration Check:")
        print(f"  Azure Endpoint: {'Set' if self.azure_endpoint else '❌ Not set'}")
        print(f"  Azure API Key: {'Set' if self.azure_api_key else '❌ Not set'}")
        print(f"  Chat Deployment: {self.chat_deployment}")
        print(f"  Embedding Deployment: {self.embedding_deployment}")
        print(f"  API Version: {self.api_version}")
        
        if not self.azure_endpoint or not self.azure_api_key:
            print("\n❌ Cannot run tests without Azure OpenAI credentials")
            return
        
        # Run tests
        self.test_basic_chains()
        self.test_conversation_memory()
        self.test_sequential_chains()
        self.test_output_parsers()
        self.test_vector_store_qa()
        self.test_agents()
        self.test_chat_prompts()
        self.test_memory_variations()
        self.test_advanced_chains()
        
        print("\n🎉 All LangChain Azure OpenAI tests completed!")

def main():
    """Main function"""
    tester = LangChainAzureOpenAITest()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 