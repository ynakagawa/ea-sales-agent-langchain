# Cosmos DB + LangChain Setup Guide

This guide will help you set up Azure Cosmos DB to work with LangChain for document storage and vector search.

## Prerequisites

1. **Azure Account**: You need an Azure subscription
2. **Python Environment**: The virtual environment we created (`env_name`)
3. **OpenAI API Key**: Already configured in our tests

## Step 1: Create Azure Cosmos DB Account

1. Go to the [Azure Portal](https://portal.azure.com)
2. Click "Create a resource"
3. Search for "Azure Cosmos DB" and select it
4. Click "Create"
5. Fill in the basic information:
   - **Subscription**: Your Azure subscription
   - **Resource Group**: Create new or use existing
   - **Account Name**: Choose a unique name (e.g., `your-langchain-cosmos`)
   - **API**: Select "Core (SQL)" or "MongoDB"
   - **Location**: Choose a region close to you
6. Click "Review + create" and then "Create"

## Step 2: Get Connection Details

Once your Cosmos DB account is created:

1. Go to your Cosmos DB account in the Azure Portal
2. In the left menu, click "Keys"
3. Copy the following values:
   - **URI** (this is your endpoint)
   - **PRIMARY KEY** (this is your access key)

## Step 3: Create Database and Container

1. In your Cosmos DB account, click "Data Explorer"
2. Click "New Database"
3. Enter database name: `langchain-db`
4. Click "OK"
5. Click "New Container"
6. Enter container name: `documents`
7. Set partition key: `/id` (or `/metadata/source` for better distribution)
8. Click "OK"

## Step 4: Set Environment Variables

Set the following environment variables in your terminal:

```bash
export COSMOS_ENDPOINT="https://your-cosmos-account.documents.azure.com:443/"
export COSMOS_KEY="your-primary-key-here"
export COSMOS_DATABASE="langchain-db"
export COSMOS_CONTAINER="documents"
```

Or create a `.env` file in your project directory:

```env
COSMOS_ENDPOINT=https://your-cosmos-account.documents.azure.com:443/
COSMOS_KEY=your-primary-key-here
COSMOS_DATABASE=langchain-db
COSMOS_CONTAINER=documents
```

## Step 5: Test the Setup

Run the test script to verify everything is working:

```bash
python test_cosmos_db.py
```

## Step 6: Run the Example

Once the tests pass, run the example script:

```bash
python cosmos_db_example.py
```

## Configuration Options

### Vector Search Index (Optional)

For better performance with vector search, you can create a vector index:

1. In your Cosmos DB container, go to "Settings" > "Indexing Policy"
2. Add a composite index for vector search:

```json
{
  "indexingMode": "consistent",
  "automatic": true,
  "includedPaths": [
    {
      "path": "/*"
    }
  ],
  "excludedPaths": [],
  "compositeIndexes": [
    [
      {
        "path": "/metadata/source",
        "order": "ascending"
      },
      {
        "path": "/metadata/type",
        "order": "ascending"
      }
    ]
  ]
}
```

### Performance Tiers

- **Provisioned throughput**: Better for consistent workloads
- **Serverless**: Better for variable workloads (pay per request)

## Usage Examples

### Basic Document Storage

```python
from cosmos_db_example import CosmosDBLangChainExample

# Initialize
example = CosmosDBLangChainExample()

# Add documents
texts = ["Your document content here"]
metadatas = [{"source": "my_docs", "type": "article"}]
example.add_documents(texts, metadatas)

# Search documents
results = example.search_documents("your search query")
```

### Advanced Search

```python
# Search with scores
results_with_scores = example.search_with_score("query", k=10)

# Filter by metadata
# (This would require custom query implementation)
```

## Troubleshooting

### Common Issues

1. **Connection Errors**: Verify your endpoint and key are correct
2. **Permission Errors**: Ensure your key has read/write permissions
3. **Rate Limiting**: Cosmos DB has request unit limits, consider scaling up
4. **Index Issues**: Vector search requires proper indexing

### Cost Optimization

- Use serverless tier for development
- Monitor request units (RUs) usage
- Consider autoscale for production workloads
- Use appropriate partition keys for even distribution

## Next Steps

1. **Production Setup**: Configure proper security and monitoring
2. **Scaling**: Set up autoscale and proper partition strategies
3. **Integration**: Connect with your LangChain applications
4. **Monitoring**: Set up Azure Monitor for performance tracking

## Resources

- [Azure Cosmos DB Documentation](https://docs.microsoft.com/en-us/azure/cosmos-db/)
- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [Azure Cosmos DB Python SDK](https://docs.microsoft.com/en-us/azure/cosmos-db/sql/sql-api-python-samples) 