#!/usr/bin/env python3
"""
Cosmos DB Setup Guide and Demo
This script shows how to properly set up Azure Cosmos DB with LangChain
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

def check_cosmos_setup():
    """Check if Cosmos DB is properly configured"""
    print("🔍 Checking Cosmos DB Configuration...")
    print("=" * 50)
    
    # Check environment variables
    endpoint = os.getenv('COSMOS_ENDPOINT')
    key = os.getenv('COSMOS_KEY')
    database = os.getenv('COSMOS_DATABASE', 'langchain-db')
    container = os.getenv('COSMOS_CONTAINER', 'documents')
    
    print(f"✅ COSMOS_ENDPOINT: {'Set' if endpoint else '❌ Not set'}")
    print(f"✅ COSMOS_KEY: {'Set' if key else '❌ Not set'}")
    print(f"✅ COSMOS_DATABASE: {database}")
    print(f"✅ COSMOS_CONTAINER: {container}")
    
    if not endpoint or not key:
        print("\n❌ Cosmos DB credentials not found!")
        print("\n📋 To set up Cosmos DB:")
        print("1. Create an Azure Cosmos DB account")
        print("2. Get your endpoint and key from the Azure portal")
        print("3. Set environment variables:")
        print("   export COSMOS_ENDPOINT='your-endpoint'")
        print("   export COSMOS_KEY='your-key'")
        print("   export COSMOS_DATABASE='your-database'")
        print("   export COSMOS_CONTAINER='your-container'")
        print("\nOr create a .env file with these variables.")
        return False
    
    # Check if credentials look like placeholders
    if "your-cosmos-account" in endpoint or "your-primary-key" in key:
        print("\n⚠️  Using placeholder credentials!")
        print("Please replace with real Cosmos DB credentials.")
        return False
    
    print("\n✅ Cosmos DB credentials found!")
    return True

def demo_cosmos_operations():
    """Demonstrate Cosmos DB operations"""
    print("\n🚀 Cosmos DB + LangChain Demo")
    print("=" * 50)
    
    # Check setup first
    if not check_cosmos_setup():
        print("\n🔄 Running in DEMO MODE")
        print("   Set real Cosmos DB credentials to perform actual operations")
        return
    
    try:
        # Initialize components
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Sample documents
        sample_texts = [
            "LangChain is a framework for developing applications powered by language models.",
            "Vector databases store and retrieve data as high-dimensional vectors.",
            "Azure Cosmos DB is a fully managed NoSQL database service.",
            "Embeddings are numerical representations of text that capture semantic meaning."
        ]
        
        sample_metadatas = [
            {"source": "langchain_docs", "topic": "framework"},
            {"source": "vector_docs", "topic": "database"},
            {"source": "azure_docs", "topic": "cloud"},
            {"source": "ml_docs", "topic": "embeddings"}
        ]
        
        # Create documents
        documents = []
        for i, text in enumerate(sample_texts):
            doc = Document(page_content=text, metadata=sample_metadatas[i])
            documents.append(doc)
        
        # Split documents
        split_docs = text_splitter.split_documents(documents)
        print(f"✅ Created {len(split_docs)} document chunks")
        
        # Create vector store
        endpoint = os.getenv('COSMOS_ENDPOINT')
        key = os.getenv('COSMOS_KEY')
        database = os.getenv('COSMOS_DATABASE', 'langchain-db')
        container = os.getenv('COSMOS_CONTAINER', 'documents')
        
        # Convert to MongoDB connection string format
        account_name = endpoint.replace('https://', '').replace('http://', '').replace('.documents.azure.com:443/', '').replace('.documents.azure.com/', '')
        connection_string = f"mongodb://{account_name}:{key}@{account_name}.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@{account_name}@"
        
        namespace = f"{database}.{container}"
        
        print(f"🔗 Connecting to Cosmos DB: {account_name}")
        vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
            connection_string=connection_string,
            namespace=namespace,
            embedding=embeddings,
            index_name="vector-index"
        )
        
        print("✅ Successfully connected to Cosmos DB!")
        
        # Add documents
        print("\n📚 Adding documents to vector store...")
        vectorstore.add_documents(split_docs)
        print(f"✅ Added {len(split_docs)} documents to Cosmos DB")
        
        # Search documents
        print("\n🔍 Searching documents...")
        search_queries = [
            "What is LangChain?",
            "How do vector databases work?",
            "Tell me about Azure services",
            "What are embeddings used for?"
        ]
        
        for query in search_queries:
            print(f"\nQuery: {query}")
            results = vectorstore.similarity_search(query, k=2)
            for i, doc in enumerate(results):
                print(f"  Result {i+1}: {doc.page_content[:100]}...")
                print(f"  Metadata: {doc.metadata}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("\nThis usually means:")
        print("1. The Cosmos DB instance doesn't exist")
        print("2. The credentials are incorrect")
        print("3. The database/container doesn't exist")
        print("4. Network connectivity issues")

def main():
    """Main function"""
    print("🔧 Cosmos DB Setup Guide")
    print("=" * 50)
    
    # Show current configuration
    check_cosmos_setup()
    
    # Run demo
    demo_cosmos_operations()
    
    print("\n📖 Next Steps:")
    print("1. Create a real Azure Cosmos DB account")
    print("2. Set up the correct credentials")
    print("3. Create the database and container")
    print("4. Run this script again")

if __name__ == "__main__":
    main() 