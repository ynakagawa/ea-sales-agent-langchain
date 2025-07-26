#!/usr/bin/env python3
"""
Simple Test Script for Cosmos DB SQL Prompt Rewrite System
"""

import os
from dotenv import load_dotenv
from cosmos_db_sql_prompt_rewrite import CosmosDBSQLPromptRewriter

def test_sql_rewrite():
    """Test the SQL rewrite functionality"""
    
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file")
        return
    
    print("🚀 Testing Cosmos DB SQL Prompt Rewrite System")
    print("=" * 50)
    
    # Initialize the rewriter
    rewriter = CosmosDBSQLPromptRewriter(openai_api_key=openai_api_key)
    
    # Test queries
    test_queries = [
        {
            "name": "Basic User Query",
            "query": """
            SELECT u.id, u.name, u.email, COUNT(p.id) as post_count
            FROM users u
            LEFT JOIN posts p ON u.id = p.user_id
            WHERE u.created_at > '2023-01-01'
            GROUP BY u.id, u.name, u.email
            HAVING COUNT(p.id) > 5
            ORDER BY post_count DESC
            """,
            "optimization_type": "performance"
        },
        {
            "name": "Cosmos DB Array Query",
            "query": """
            SELECT c.id, c.name, 
                   ARRAY_LENGTH(c.posts) as postCount,
                   c.posts[0].title as latestPost
            FROM c
            WHERE ARRAY_LENGTH(c.posts) > 0
            AND c.posts[0].createdAt > '2023-01-01T00:00:00Z'
            """,
            "optimization_type": "cosmos_db"
        },
        {
            "name": "Complex Join Query",
            "query": """
            SELECT u.id, u.name, u.email,
                   COUNT(p.id) as postCount,
                   AVG(p.likes) as avgLikes
            FROM users u
            JOIN posts p ON u.id = p.userId
            WHERE u.createdAt > '2023-01-01T00:00:00Z'
            AND p.status = 'published'
            GROUP BY u.id, u.name, u.email
            HAVING COUNT(p.id) > 5
            ORDER BY postCount DESC
            """,
            "optimization_type": "readability"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        print(f"Optimization Type: {test_case['optimization_type'].upper()}")
        print(f"Original Query: {test_case['query'].strip()}")
        
        try:
            result = rewriter.rewrite_sql_query(
                test_case['query'].strip(),
                optimization_type=test_case['optimization_type']
            )
            
            print(f"\n✅ Rewritten Query:")
            print(result.rewritten_query)
            print(f"\n📝 Explanation: {result.explanation[:200]}...")
            print(f"🎯 Confidence Score: {result.confidence_score:.2f}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 50)
    
    print("\n✨ Test completed!")

if __name__ == "__main__":
    test_sql_rewrite() 