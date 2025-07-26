#!/usr/bin/env python3
"""
Azure OpenAI Service Test Script
Tests Azure OpenAI Service integration with LangChain
"""

import os
import warnings
from typing import List, Dict, Any
from langchain_community.llms import AzureOpenAI
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

class AzureOpenAITest:
    """Test class for Azure OpenAI Service integration"""
    
    def __init__(self):
        """Initialize Azure OpenAI test components"""
        # Get Azure OpenAI credentials
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-35-turbo')
        self.chat_deployment = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'gpt-35-turbo')
        self.embedding_deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        
        # Check if credentials are available
        if not self.azure_endpoint or not self.azure_api_key:
            print("⚠️  Azure OpenAI credentials not found!")
            print("Please set the following environment variables:")
            print("  - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint")
            print("  - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key")
            print("  - AZURE_OPENAI_DEPLOYMENT_NAME: Model deployment name (optional)")
            print("  - AZURE_OPENAI_CHAT_DEPLOYMENT: Chat model deployment (optional)")
            print("  - AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Embedding model deployment (optional)")
            print("  - AZURE_OPENAI_API_VERSION: API version (optional)")
            return
        
        # Initialize components
        self.llm = None
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM, Chat, and embeddings components"""
        try:
            # Initialize Azure OpenAI LLM (completion model)
            self.llm = AzureOpenAI(
                azure_deployment=self.deployment_name,
                openai_api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                temperature=0.1
            )
            print(f"✅ Initialized Azure OpenAI LLM with deployment: {self.deployment_name}")
            
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
    
    def test_completion_model(self):
        """Test Azure OpenAI completion model"""
        print("\n🧠 Testing Azure OpenAI Completion Model")
        print("=" * 50)
        
        if not self.llm:
            print("❌ LLM not initialized")
            return
        
        try:
            # Test simple completion
            prompt = "Explain what Azure OpenAI Service is in one sentence."
            response = self.llm(prompt)
            print(f"✅ Simple completion: {response}")
            
            # Test with different temperatures
            temperatures = [0.1, 0.5, 0.9]
            for temp in temperatures:
                self.llm.temperature = temp
                response = self.llm("Write a creative story about AI in 2 sentences.")
                print(f"✅ Temperature {temp}: {response[:100]}...")
            
            # Reset temperature
            self.llm.temperature = 0.1
            
        except Exception as e:
            print(f"❌ Completion model test failed: {e}")
    
    def test_chat_model(self):
        """Test Azure OpenAI chat model"""
        print("\n💬 Testing Azure OpenAI Chat Model")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Test simple chat
            messages = [
                HumanMessage(content="Hello! What can you help me with?")
            ]
            response = self.chat_model(messages)
            print(f"✅ Simple chat: {response.content}")
            
            # Test with system message
            messages = [
                SystemMessage(content="You are a helpful AI assistant specialized in Azure services."),
                HumanMessage(content="Tell me about Azure OpenAI Service.")
            ]
            response = self.chat_model(messages)
            print(f"✅ Chat with system message: {response.content[:200]}...")
            
            # Test conversation chain
            conversation = ConversationChain(
                llm=self.chat_model,
                memory=ConversationBufferMemory()
            )
            
            response1 = conversation.predict(input="What is machine learning?")
            print(f"✅ Conversation 1: {response1[:150]}...")
            
            response2 = conversation.predict(input="How does it relate to Azure?")
            print(f"✅ Conversation 2: {response2[:150]}...")
            
        except Exception as e:
            print(f"❌ Chat model test failed: {e}")
    
    def test_embeddings(self):
        """Test Azure OpenAI embeddings"""
        print("\n🔢 Testing Azure OpenAI Embeddings")
        print("=" * 50)
        
        if not self.embeddings:
            print("❌ Embeddings not initialized")
            return
        
        try:
            # Test single text embedding
            text = "Azure OpenAI Service provides enterprise-grade access to OpenAI models."
            embedding = self.embeddings.embed_query(text)
            print(f"✅ Single embedding: {len(embedding)} dimensions")
            print(f"   Sample values: {embedding[:5]}...")
            
            # Test multiple texts
            texts = [
                "Azure OpenAI Service is a fully managed service.",
                "It provides secure access to OpenAI's powerful language models.",
                "Enterprise customers can deploy and manage AI models safely."
            ]
            
            embeddings = self.embeddings.embed_documents(texts)
            print(f"\n✅ Multiple embeddings: {len(embeddings)} documents")
            for i, emb in enumerate(embeddings):
                print(f"   Document {i+1}: {len(emb)} dimensions")
            
            # Test similarity
            query = "What is Azure OpenAI?"
            query_embedding = self.embeddings.embed_query(query)
            
            import numpy as np
            similarities = []
            for i, doc_emb in enumerate(embeddings):
                similarity = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
                similarities.append((similarity, texts[i]))
            
            similarities.sort(reverse=True)
            print(f"\n✅ Similarity search results:")
            for i, (sim, text) in enumerate(similarities[:2]):
                print(f"   {i+1}. {text[:50]}... (similarity: {sim:.3f})")
            
        except Exception as e:
            print(f"❌ Embeddings test failed: {e}")
    
    def test_prompt_templates(self):
        """Test prompt templates with Azure OpenAI"""
        print("\n📝 Testing Prompt Templates")
        print("=" * 50)
        
        if not self.llm:
            print("❌ LLM not initialized")
            return
        
        try:
            # Test basic prompt template
            template = """
            You are an Azure expert. Please explain {service} in detail.
            Focus on:
            1. What it is
            2. Key features
            3. Use cases
            
            Keep your response under 200 words.
            """
            
            prompt = PromptTemplate(
                input_variables=["service"],
                template=template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            services = ["Azure OpenAI Service", "Azure Cognitive Services", "Azure Machine Learning"]
            for service in services:
                response = chain.run(service=service)
                print(f"\n📝 {service}:")
                print(f"   {response[:150]}...")
            
        except Exception as e:
            print(f"❌ Prompt templates test failed: {e}")
    
    def test_chat_prompts(self):
        """Test chat prompt templates"""
        print("\n💬 Testing Chat Prompt Templates")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Test chat prompt template
            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant specialized in {domain}."),
                ("human", "Explain {concept} in simple terms.")
            ])
            
            chain = LLMChain(llm=self.chat_model, prompt=template)
            
            domains_and_concepts = [
                ("cloud computing", "serverless architecture"),
                ("machine learning", "neural networks"),
                ("data science", "data preprocessing")
            ]
            
            for domain, concept in domains_and_concepts:
                response = chain.run(domain=domain, concept=concept)
                print(f"\n📝 {domain} - {concept}:")
                print(f"   {response[:150]}...")
            
        except Exception as e:
            print(f"❌ Chat prompts test failed: {e}")
    
    def test_document_processing(self):
        """Test document processing with Azure OpenAI"""
        print("\n📄 Testing Document Processing")
        print("=" * 50)
        
        if not self.embeddings:
            print("❌ Embeddings not initialized")
            return
        
        try:
            # Sample documents
            documents = [
                Document(
                    page_content="Azure OpenAI Service provides enterprise-grade access to OpenAI's powerful language models with built-in security and compliance features.",
                    metadata={"source": "azure_docs", "topic": "ai_service", "type": "overview"}
                ),
                Document(
                    page_content="The service supports various models including GPT-3.5, GPT-4, and embedding models, all deployed securely in your Azure environment.",
                    metadata={"source": "azure_docs", "topic": "models", "type": "technical"}
                ),
                Document(
                    page_content="Enterprise customers can use Azure OpenAI Service for applications like chatbots, content generation, and data analysis with full control over their data.",
                    metadata={"source": "azure_docs", "topic": "use_cases", "type": "applications"}
                )
            ]
            
            print(f"✅ Created {len(documents)} documents")
            
            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            print(f"✅ Split into {len(split_docs)} chunks")
            
            # Generate embeddings for chunks
            texts = [doc.page_content for doc in split_docs]
            embeddings = self.embeddings.embed_documents(texts)
            
            print(f"✅ Generated embeddings for {len(embeddings)} chunks")
            
            # Show sample
            for i, (doc, emb) in enumerate(zip(split_docs[:2], embeddings[:2])):
                print(f"\n📝 Chunk {i+1}:")
                print(f"   Content: {doc.page_content[:80]}...")
                print(f"   Metadata: {doc.metadata}")
                print(f"   Embedding: {len(emb)} dimensions")
            
        except Exception as e:
            print(f"❌ Document processing test failed: {e}")
    
    def test_advanced_features(self):
        """Test advanced Azure OpenAI features"""
        print("\n🚀 Testing Advanced Features")
        print("=" * 50)
        
        if not self.chat_model:
            print("❌ Chat model not initialized")
            return
        
        try:
            # Test function calling simulation
            messages = [
                SystemMessage(content="You are a helpful assistant that can analyze text and extract structured information."),
                HumanMessage(content="Analyze this text and extract key information: 'Azure OpenAI Service costs $0.002 per 1K tokens and supports GPT-4 models.'")
            ]
            
            response = self.chat_model(messages)
            print(f"✅ Text analysis: {response.content[:200]}...")
            
            # Test creative writing
            messages = [
                SystemMessage(content="You are a creative writer. Write engaging content."),
                HumanMessage(content="Write a short story about an AI assistant helping a developer build an application.")
            ]
            
            response = self.chat_model(messages)
            print(f"✅ Creative writing: {response.content[:200]}...")
            
            # Test code generation
            messages = [
                SystemMessage(content="You are a Python expert. Write clean, well-documented code."),
                HumanMessage(content="Write a Python function to calculate the cosine similarity between two vectors.")
            ]
            
            response = self.chat_model(messages)
            print(f"✅ Code generation: {response.content[:200]}...")
            
        except Exception as e:
            print(f"❌ Advanced features test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n⚠️ Testing Error Handling")
        print("=" * 50)
        
        if not self.llm:
            print("❌ LLM not initialized")
            return
        
        try:
            # Test empty input
            try:
                response = self.llm("")
                print(f"✅ Empty input handled: {response[:50]}...")
            except Exception as e:
                print(f"✅ Empty input error caught: {type(e).__name__}")
            
            # Test very long input
            long_text = "This is a very long text. " * 1000
            try:
                response = self.llm(long_text)
                print(f"✅ Long input handled: {response[:50]}...")
            except Exception as e:
                print(f"✅ Long input error caught: {type(e).__name__}")
            
            # Test special characters
            special_text = "Test with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
            try:
                response = self.llm(special_text)
                print(f"✅ Special characters handled: {response[:50]}...")
            except Exception as e:
                print(f"✅ Special characters error caught: {type(e).__name__}")
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Azure OpenAI Service Integration Test")
        print("=" * 60)
        
        # Check configuration
        print("🔍 Configuration Check:")
        print(f"  Azure Endpoint: {'Set' if self.azure_endpoint else '❌ Not set'}")
        print(f"  Azure API Key: {'Set' if self.azure_api_key else '❌ Not set'}")
        print(f"  LLM Deployment: {self.deployment_name}")
        print(f"  Chat Deployment: {self.chat_deployment}")
        print(f"  Embedding Deployment: {self.embedding_deployment}")
        print(f"  API Version: {self.api_version}")
        
        if not self.azure_endpoint or not self.azure_api_key:
            print("\n❌ Cannot run tests without Azure OpenAI credentials")
            return
        
        # Run tests
        self.test_completion_model()
        self.test_chat_model()
        self.test_embeddings()
        self.test_prompt_templates()
        self.test_chat_prompts()
        self.test_document_processing()
        self.test_advanced_features()
        self.test_error_handling()
        
        print("\n🎉 All Azure OpenAI tests completed!")

def main():
    """Main function"""
    tester = AzureOpenAITest()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 