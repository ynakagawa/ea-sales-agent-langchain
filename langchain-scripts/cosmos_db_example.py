#!/usr/bin/env python3
"""
Example script demonstrating Cosmos DB integration with LangChain
"""

import os
import warnings
from typing import List, Dict, Any
from langchain_community.vectorstores import AzureCosmosDBVectorSearch
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

# Suppress CosmosDB cluster warnings
warnings.filterwarnings("ignore", message="You appear to be connected to a CosmosDB cluster")

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)


class CosmosDBLangChainExample:
    def __init__(self):
        """Initialize the Cosmos DB LangChain example"""
        # Try to get from environment variables first
        self.endpoint = os.getenv('COSMOS_ENDPOINT')
        self.key = os.getenv('COSMOS_KEY')
        self.database_name = os.getenv('COSMOS_DATABASE', 'langchain-db')
        self.container_name = os.getenv('COSMOS_CONTAINER', 'documents')
        
        # If not set, use example values for demonstration
        if not self.endpoint or not self.key:
            print("⚠️  Using example connection details for demonstration")
            print("   Set COSMOS_ENDPOINT and COSMOS_KEY environment variables for real usage")
            self.endpoint = "https://your-cosmos-account.documents.azure.com:443/"
            self.key = "your-primary-key-here"
            self.database_name = "ea-langchain-cosmos"
            self.container_name = "documents"
        
        self.openai_api_key = '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'
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def check_credentials(self) -> bool:
        """Check if Cosmos DB credentials are available"""
        if not self.endpoint or not self.key or self.endpoint == "https://your-cosmos-account.documents.azure.com:443/" or self.key == "your-primary-key-here":
            print("❌ Cosmos DB credentials not found!")
            print("Please set the following environment variables:")
            print("  - COSMOS_ENDPOINT: Your Cosmos DB endpoint URL")
            print("  - COSMOS_KEY: Your Cosmos DB access key")
            print("  - COSMOS_DATABASE: Database name (optional, defaults to 'langchain-db')")
            print("  - COSMOS_CONTAINER: Container name (optional, defaults to 'documents')")
            print("\nOr create a .env file with these variables for automatic loading.")
            return False
        return True
    
    def is_demo_mode(self) -> bool:
        """Check if running in demo mode with placeholder credentials"""
        # Check for obvious placeholder values
        is_demo = (not self.endpoint or not self.key or 
                self.endpoint == "https://your-cosmos-account.documents.azure.com:443/" or 
                self.key == "your-primary-key-here" or
                "your-cosmos-account" in self.endpoint or
                "your-primary-key" in self.key)
        
        return is_demo
    
    def create_vectorstore(self) -> AzureCosmosDBVectorSearch:
        """Create and return a Cosmos DB vector store"""
        if self.is_demo_mode():
            print("🔄 Demo mode: Skipping vector store creation")
            raise ValueError("Demo mode - no real database connection")
        
        if not self.check_credentials():
            raise ValueError("Cosmos DB credentials not available")
        
        # For Cosmos DB with MongoDB API, use MongoDB connection string format
        if "mongodb" in self.endpoint.lower():
            # MongoDB API format
            connection_string = f"mongodb://{self.endpoint}:{self.key}@{self.endpoint.replace('https://', '').replace('http://', '')}:10255/?ssl=true&replicaSet=globaldb"
        else:
            # SQL API format - but AzureCosmosDBVectorSearch expects MongoDB format
            # Convert to MongoDB format for vector search
            account_name = self.endpoint.replace('https://', '').replace('http://', '').replace('.documents.azure.com:443/', '').replace('.documents.azure.com/', '')
            connection_string = f"mongodb://{account_name}:{self.key}@{account_name}.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@{account_name}@"
        
        namespace = f"{self.database_name}.{self.container_name}"
        
        vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
            connection_string=connection_string,
            namespace=namespace,
            embedding=self.embeddings,
            index_name="vector-index"
        )
        
        print(f"✅ Created vector store for database: {self.database_name}, container: {self.container_name}")
        return vectorstore
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """Add documents to the vector store"""
        if self.is_demo_mode():
            print("🔄 Demo mode: Simulating document addition...")
            print(f"   Would add {len(texts)} documents to Cosmos DB")
            print("   Set real COSMOS_ENDPOINT and COSMOS_KEY to perform actual operations")
            return
        
        try:
            vectorstore = self.create_vectorstore()
            
            # Create documents
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
            
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            vectorstore.add_documents(split_docs)
            
            print(f"✅ Successfully added {len(split_docs)} document chunks to Cosmos DB")
            
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search for documents using similarity search"""
        if self.is_demo_mode():
            print("🔄 Demo mode: Simulating document search...")
            print(f"   Would search for: '{query}' (k={k})")
            print("   Set real COSMOS_ENDPOINT and COSMOS_KEY to perform actual operations")
            return []
        
        try:
            vectorstore = self.create_vectorstore()
            
            # Perform similarity search
            results = vectorstore.similarity_search(query, k=k)
            
            print(f"✅ Found {len(results)} documents for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"❌ Failed to search documents: {e}")
            return []
    
    def search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search for documents with similarity scores"""
        if self.is_demo_mode():
            print("🔄 Demo mode: Simulating scored search...")
            print(f"   Would search for: '{query}' with scores (k={k})")
            print("   Set real COSMOS_ENDPOINT and COSMOS_KEY to perform actual operations")
            return []
        
        try:
            vectorstore = self.create_vectorstore()
            
            # Perform similarity search with scores
            results = vectorstore.similarity_search_with_score(query, k=k)
            
            print(f"✅ Found {len(results)} documents with scores for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"❌ Failed to search documents with scores: {e}")
            return []

def main():
    """Main example function"""
    print("🚀 Cosmos DB + LangChain Example")
    print("=" * 50)
    
    # Create example instance
    example = CosmosDBLangChainExample()
    
    # Check if running in demo mode
    if example.is_demo_mode():
        print("🔄 Running in DEMO MODE")
        print("   Set COSMOS_ENDPOINT and COSMOS_KEY environment variables for real operations")
        print()
    elif not example.check_credentials():
        print("\n📝 Example usage:")
        print("1. Set your Cosmos DB credentials as environment variables")
        print("2. Run this script to test document operations")
        print("3. Use the methods to add and search documents")
        return
    
    # Sample documents
    sample_texts = [
        "LangChain is a framework for developing applications powered by language models. It provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.",
        "Vector databases store and retrieve data as high-dimensional vectors, which are mathematical representations of features or attributes. They are commonly used for similarity search and machine learning applications.",
        "Azure Cosmos DB is a fully managed NoSQL database service for modern app development. It offers single-digit millisecond response times, automatic and instant scalability, and guaranteed availability.",
        "Embeddings are numerical representations of text that capture semantic meaning. They allow us to compare documents and find similar content using mathematical operations."
    ]
    
    sample_metadatas = [
        {"source": "langchain_docs", "topic": "framework", "type": "introduction"},
        {"source": "vector_docs", "topic": "database", "type": "concept"},
        {"source": "azure_docs", "topic": "cloud", "type": "service"},
        {"source": "ml_docs", "topic": "embeddings", "type": "concept"}
    ]
    
    print("\n📚 Adding sample documents...")
    example.add_documents(sample_texts, sample_metadatas)
    
    print("\n🔍 Searching for documents...")
    search_queries = [
        "What is LangChain?",
        "How do vector databases work?",
        "Tell me about Azure services",
        "What are embeddings used for?"
    ]
    
    for query in search_queries:
        print(f"\nQuery: {query}")
        results = example.search_documents(query, k=2)
        if results:  # Only show results if we have them (not in demo mode)
            for i, doc in enumerate(results):
                print(f"  Result {i+1}: {doc.page_content[:100]}...")
                print(f"  Metadata: {doc.metadata}")
    
    print("\n🎉 Example completed!")

if __name__ == "__main__":
    main() 