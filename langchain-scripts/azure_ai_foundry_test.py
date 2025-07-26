#!/usr/bin/env python3
"""
Azure AI Foundry Test Script
Tests Azure AI Foundry integration with LangChain
"""

import os
import warnings
from typing import List, Dict, Any
from langchain_community.llms import AzureOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

class AzureAIFoundryTest:
    """Test class for Azure AI Foundry integration"""
    
    def __init__(self):
        """Initialize Azure AI Foundry test components"""
        # Get Azure AI Foundry credentials
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-35-turbo')
        self.embedding_deployment = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        
        # Check if credentials are available
        if not self.azure_endpoint or not self.azure_api_key:
            print("⚠️  Azure AI Foundry credentials not found!")
            print("Please set the following environment variables:")
            print("  - AZURE_OPENAI_ENDPOINT: Your Azure AI Foundry endpoint")
            print("  - AZURE_OPENAI_API_KEY: Your Azure AI Foundry API key")
            print("  - AZURE_OPENAI_DEPLOYMENT_NAME: Model deployment name (optional)")
            print("  - AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Embedding model deployment (optional)")
            print("  - AZURE_OPENAI_API_VERSION: API version (optional)")
            return
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM and embeddings components"""
        try:
            # Initialize Azure OpenAI LLM
            self.llm = AzureOpenAI(
                azure_deployment=self.deployment_name,
                openai_api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                temperature=0.1
            )
            print(f"✅ Initialized Azure OpenAI LLM with deployment: {self.deployment_name}")
            
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
    
    def test_llm_generation(self):
        """Test LLM text generation"""
        print("\n🧠 Testing LLM Text Generation")
        print("=" * 50)
        
        if not self.llm:
            print("❌ LLM not initialized")
            return
        
        try:
            # Test simple generation
            prompt = "Explain what Azure AI Foundry is in one sentence."
            response = self.llm(prompt)
            print(f"✅ Simple generation: {response}")
            
            # Test with prompt template
            template = """
            You are a helpful AI assistant. Please provide a brief explanation of {topic}.
            Keep your response under 100 words.
            """
            
            prompt_template = PromptTemplate(
                input_variables=["topic"],
                template=template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            
            topics = ["machine learning", "vector databases", "LangChain"]
            for topic in topics:
                response = chain.run(topic=topic)
                print(f"\n📝 {topic}: {response}")
            
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
    
    def test_embeddings(self):
        """Test embedding generation"""
        print("\n🔢 Testing Embedding Generation")
        print("=" * 50)
        
        if not self.embeddings:
            print("❌ Embeddings not initialized")
            return
        
        try:
            # Test single text embedding
            text = "Azure AI Foundry is a comprehensive platform for building AI applications."
            embedding = self.embeddings.embed_query(text)
            print(f"✅ Single embedding: {len(embedding)} dimensions")
            print(f"   Sample values: {embedding[:5]}...")
            
            # Test multiple texts
            texts = [
                "Machine learning is a subset of artificial intelligence.",
                "Vector databases store high-dimensional data efficiently.",
                "LangChain provides tools for building LLM applications."
            ]
            
            embeddings = self.embeddings.embed_documents(texts)
            print(f"\n✅ Multiple embeddings: {len(embeddings)} documents")
            for i, emb in enumerate(embeddings):
                print(f"   Document {i+1}: {len(emb)} dimensions")
            
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
    
    def test_document_processing(self):
        """Test document processing with embeddings"""
        print("\n📄 Testing Document Processing")
        print("=" * 50)
        
        if not self.embeddings:
            print("❌ Embeddings not initialized")
            return
        
        try:
            # Sample documents
            documents = [
                Document(
                    page_content="Azure AI Foundry provides enterprise-grade AI capabilities with built-in security and compliance features.",
                    metadata={"source": "azure_docs", "topic": "ai_platform"}
                ),
                Document(
                    page_content="Vector databases enable efficient similarity search and retrieval of high-dimensional data representations.",
                    metadata={"source": "vector_docs", "topic": "databases"}
                ),
                Document(
                    page_content="LangChain offers a comprehensive framework for developing applications powered by language models.",
                    metadata={"source": "langchain_docs", "topic": "framework"}
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
            print(f"❌ Document processing failed: {e}")
    
    def test_similarity_search(self):
        """Test similarity search with embeddings"""
        print("\n🔍 Testing Similarity Search")
        print("=" * 50)
        
        if not self.embeddings:
            print("❌ Embeddings not initialized")
            return
        
        try:
            # Sample documents
            documents = [
                "Azure AI Foundry provides enterprise AI capabilities",
                "Machine learning models require training data",
                "Vector databases store embeddings efficiently",
                "LangChain simplifies LLM application development",
                "Natural language processing enables text understanding"
            ]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(documents)
            
            # Test queries
            queries = [
                "What is Azure AI Foundry?",
                "How do vector databases work?",
                "Tell me about machine learning"
            ]
            
            for query in queries:
                print(f"\n🔍 Query: {query}")
                query_embedding = self.embeddings.embed_query(query)
                
                # Simple similarity calculation (cosine similarity)
                similarities = []
                for i, doc_emb in enumerate(embeddings):
                    # Calculate cosine similarity
                    import numpy as np
                    similarity = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
                    similarities.append((similarity, documents[i]))
                
                # Sort by similarity
                similarities.sort(reverse=True)
                
                print("   Most similar documents:")
                for i, (sim, doc) in enumerate(similarities[:3]):
                    print(f"     {i+1}. {doc} (similarity: {sim:.3f})")
            
        except Exception as e:
            print(f"❌ Similarity search failed: {e}")
    
    def test_chain_operations(self):
        """Test LangChain operations with Azure AI Foundry"""
        print("\n⛓️ Testing LangChain Chain Operations")
        print("=" * 50)
        
        if not self.llm:
            print("❌ LLM not initialized")
            return
        
        try:
            # Create a simple chain
            template = """
            You are an AI expert. Analyze the following text and provide insights:
            
            Text: {text}
            
            Please provide:
            1. Key topics mentioned
            2. Technical concepts identified
            3. Potential applications
            
            Keep your response concise and structured.
            """
            
            prompt = PromptTemplate(
                input_variables=["text"],
                template=template
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Test with different texts
            test_texts = [
                "Azure AI Foundry combines machine learning, vector databases, and natural language processing to create comprehensive AI solutions.",
                "Vector databases enable efficient storage and retrieval of high-dimensional embeddings for similarity search applications."
            ]
            
            for i, text in enumerate(test_texts):
                print(f"\n📝 Analysis {i+1}:")
                print(f"Input: {text[:100]}...")
                response = chain.run(text=text)
                print(f"Analysis: {response}")
            
        except Exception as e:
            print(f"❌ Chain operations failed: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Azure AI Foundry Integration Test")
        print("=" * 60)
        
        # Check configuration
        print("🔍 Configuration Check:")
        print(f"  Azure Endpoint: {'Set' if self.azure_endpoint else '❌ Not set'}")
        print(f"  Azure API Key: {'Set' if self.azure_api_key else '❌ Not set'}")
        print(f"  LLM Deployment: {self.deployment_name}")
        print(f"  Embedding Deployment: {self.embedding_deployment}")
        print(f"  API Version: {self.api_version}")
        
        if not self.azure_endpoint or not self.azure_api_key:
            print("\n❌ Cannot run tests without Azure AI Foundry credentials")
            return
        
        # Run tests
        self.test_llm_generation()
        self.test_embeddings()
        self.test_document_processing()
        self.test_similarity_search()
        self.test_chain_operations()
        
        print("\n🎉 All tests completed!")

def main():
    """Main function"""
    tester = AzureAIFoundryTest()
    tester.run_all_tests()

if __name__ == "__main__":
    main() 