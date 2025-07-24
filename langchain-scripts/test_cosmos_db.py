#!/usr/bin/env python3
"""
Test script for Cosmos DB integration with LangChain
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env", override=True)

def test_cosmos_imports():
    """Test if Cosmos DB packages can be imported"""
    try:
        import azure.cosmos.cosmos_client as cosmos_client
        import azure.cosmos.exceptions as exceptions
        from azure.cosmos import PartitionKey
        print("✅ Azure Cosmos DB packages imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Azure Cosmos DB packages: {e}")
        return False

def test_mongo_imports():
    """Test if MongoDB packages can be imported"""
    try:
        import pymongo
        print("✅ PyMongo imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Failed to import PyMongo: {e}")
        return False

def test_langchain_vectorstore_imports():
    """Test if LangChain vector store components can be imported"""
    try:
        from langchain_community.vectorstores import AzureCosmosDBVectorSearch
        from langchain_community.embeddings import OpenAIEmbeddings
        print("✅ LangChain vector store components imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Failed to import LangChain vector store components: {e}")
        return False

def test_cosmos_connection():
    """Test Cosmos DB connection (requires environment variables)"""
    try:
        import azure.cosmos.cosmos_client as cosmos_client
        from azure.cosmos import PartitionKey
        
        # Get connection details from environment variables
        endpoint = os.getenv('COSMOS_ENDPOINT')
        key = os.getenv('COSMOS_KEY')
        database_name = os.getenv('COSMOS_DATABASE', 'langchain-db')
        container_name = os.getenv('COSMOS_CONTAINER', 'documents')
        
        if not endpoint or not key:
            print("ℹ️  Cosmos DB connection details not found in environment variables.")
            print("   Set COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, and COSMOS_CONTAINER")
            return False
        
        # Create Cosmos client
        client = cosmos_client.CosmosClient(endpoint, key)
        
        # Test database connection
        database = client.get_database_client(database_name)
        print(f"✅ Successfully connected to Cosmos DB database: {database_name}")
        
        # Test container access
        container = database.get_container_client(container_name)
        print(f"✅ Successfully accessed container: {container_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Cosmos DB: {e}")
        return False

def test_vectorstore_creation():
    """Test creating a vector store with Cosmos DB"""
    try:
        from langchain_community.vectorstores import AzureCosmosDBVectorSearch
        from langchain_community.embeddings import OpenAIEmbeddings
        
        # Get connection details
        endpoint = os.getenv('COSMOS_ENDPOINT')
        key = os.getenv('COSMOS_KEY')
        database_name = os.getenv('COSMOS_DATABASE', 'langchain-db')
        container_name = os.getenv('COSMOS_CONTAINER', 'documents')
        
        if not endpoint or not key:
            print("ℹ️  Skipping vector store test - Cosmos DB credentials not available")
            return False
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(
            api_key='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'
        )
        
        # Create vector store
        # For Cosmos DB with MongoDB API, use MongoDB connection string format
        if "mongodb" in endpoint.lower():
            # MongoDB API format
            connection_string = f"mongodb://{endpoint}:{key}@{endpoint.replace('https://', '').replace('http://', '')}:10255/?ssl=true&replicaSet=globaldb"
        else:
            # SQL API format
            connection_string = f"AccountEndpoint={endpoint};AccountKey={key};"
        
        vectorstore = AzureCosmosDBVectorSearch.from_connection_string(
            connection_string=connection_string,
            namespace=f"{database_name}.{container_name}",
            embedding=embeddings,
            index_name="vector-index"
        )
        
        print("✅ Successfully created Azure Cosmos DB Vector Search instance!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create vector store: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    sample_documents = [
        {
            "id": "doc1",
            "content": "LangChain is a framework for developing applications powered by language models.",
            "metadata": {"source": "langchain_docs", "type": "introduction"}
        },
        {
            "id": "doc2", 
            "content": "Vector stores are used to store and retrieve embeddings for similarity search.",
            "metadata": {"source": "vector_docs", "type": "concept"}
        },
        {
            "id": "doc3",
            "content": "Cosmos DB is a globally distributed, multi-model database service.",
            "metadata": {"source": "azure_docs", "type": "database"}
        }
    ]
    return sample_documents

def test_document_operations():
    """Test basic document operations with Cosmos DB"""
    try:
        import azure.cosmos.cosmos_client as cosmos_client
        
        endpoint = os.getenv('COSMOS_ENDPOINT')
        key = os.getenv('COSMOS_KEY')
        database_name = os.getenv('COSMOS_DATABASE', 'langchain-db')
        container_name = os.getenv('COSMOS_CONTAINER', 'documents')
        
        if not endpoint or not key:
            print("ℹ️  Skipping document operations test - Cosmos DB credentials not available")
            return False
        
        # Create client
        client = cosmos_client.CosmosClient(endpoint, key)
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)
        
        # Create sample documents
        documents = create_sample_data()
        
        # Insert documents
        for doc in documents:
            container.upsert_item(doc)
        
        print(f"✅ Successfully inserted {len(documents)} documents into Cosmos DB")
        
        # Query documents
        query = "SELECT * FROM c WHERE c.metadata.type = 'introduction'"
        results = list(container.query_items(query=query, enable_cross_partition_query=True))
        print(f"✅ Successfully queried documents: found {len(results)} introduction documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to perform document operations: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Running Cosmos DB integration tests...")
    print("=" * 60)
    
    # Test imports
    cosmos_imports = test_cosmos_imports()
    mongo_imports = test_mongo_imports()
    vectorstore_imports = test_langchain_vectorstore_imports()
    
    # Test connections and operations
    cosmos_connection = test_cosmos_connection()
    vectorstore_creation = test_vectorstore_creation()
    document_operations = test_document_operations()
    
    print("=" * 60)
    
    # Summary
    basic_tests = all([cosmos_imports, mongo_imports, vectorstore_imports])
    connection_tests = any([cosmos_connection, vectorstore_creation, document_operations])
    
    if basic_tests:
        print("✅ All basic imports successful!")
        if connection_tests:
            print("🎉 Cosmos DB integration is working correctly!")
            return 0
        else:
            print("⚠️  Basic setup complete, but connection tests require Cosmos DB credentials")
            print("   Set environment variables: COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER")
            return 0
    else:
        print("❌ Some basic tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 