#!/usr/bin/env python3
"""
Cosmos DB SQL Prompt Rewrite Demo
This script demonstrates how to use the SQL prompt rewrite system with Cosmos DB
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from cosmos_db_sql_prompt_rewrite import CosmosDBSQLPromptRewriter, SQLRewriteResult

class CosmosDBSQLRewriteDemo:
    """Demo class for Cosmos DB SQL prompt rewriting"""
    
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Sample Cosmos DB SQL queries for demonstration
        self.cosmos_db_queries = [
            # Basic query
            """
            SELECT c.id, c.name, c.email, c.createdAt
            FROM c
            WHERE c.type = 'user' AND c.isActive = true
            ORDER BY c.createdAt DESC
            """,
            
            # Query with array operations
            """
            SELECT c.id, c.name, 
                   ARRAY_LENGTH(c.posts) as postCount,
                   c.posts[0].title as latestPost
            FROM c
            WHERE ARRAY_LENGTH(c.posts) > 0
            AND c.posts[0].createdAt > '2023-01-01T00:00:00Z'
            """,
            
            # Complex query with joins (simulated)
            """
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
            
            # Query with JSON path expressions
            """
            SELECT c.id, c.name,
                   c.profile.address.city,
                   c.profile.preferences.theme
            FROM c
            WHERE c.profile.address.country = 'US'
            AND c.profile.preferences.notifications = true
            """,
            
            # Subquery example
            """
            SELECT c.id, c.name, c.email
            FROM c
            WHERE c.id IN (
                SELECT DISTINCT p.userId
                FROM posts p
                WHERE p.createdAt > '2023-01-01T00:00:00Z'
                AND p.likes > 100
            )
            """
        ]
        
        # Initialize the rewriter
        if self.openai_api_key:
            self.rewriter = CosmosDBSQLPromptRewriter(openai_api_key=self.openai_api_key)
        else:
            self.rewriter = None
    
    def show_cosmos_db_sql_rewrite_overview(self):
        """Show overview of Cosmos DB SQL rewrite capabilities"""
        print("🚀 Cosmos DB SQL Prompt Rewrite System Overview")
        print("=" * 60)
        print()
        print("📋 Key Features:")
        print("  • SQL query optimization for Cosmos DB")
        print("  • Performance-focused rewrites")
        print("  • Readability improvements")
        print("  • Cosmos DB specific optimizations")
        print("  • Batch processing capabilities")
        print("  • Query similarity search")
        print("  • Pattern analysis")
        print()
        print("🎯 Optimization Types:")
        print("  • general: General query improvements")
        print("  • performance: Performance-focused optimizations")
        print("  • readability: Code readability improvements")
        print("  • cosmos_db: Cosmos DB specific optimizations")
        print()
    
    def demo_basic_rewrite(self):
        """Demonstrate basic SQL query rewriting"""
        print("🔄 Basic SQL Query Rewrite Demo")
        print("-" * 40)
        
        if not self.rewriter:
            print("❌ OpenAI API key not available. Running in demo mode.")
            return
        
        sample_query = """
        SELECT u.id, u.name, u.email, COUNT(p.id) as post_count
        FROM users u
        LEFT JOIN posts p ON u.id = p.user_id
        WHERE u.created_at > '2023-01-01'
        GROUP BY u.id, u.name, u.email
        HAVING COUNT(p.id) > 5
        ORDER BY post_count DESC
        """
        
        print("Original Query:")
        print(sample_query.strip())
        print()
        
        # Test different optimization types
        optimization_types = ["general", "performance", "readability", "cosmos_db"]
        
        for opt_type in optimization_types:
            print(f"🔧 {opt_type.upper()} Optimization:")
            try:
                result = self.rewriter.rewrite_sql_query(sample_query, optimization_type=opt_type)
                
                print(f"Rewritten Query:")
                print(result.rewritten_query)
                print(f"Explanation: {result.explanation[:150]}...")
                print(f"Confidence: {result.confidence_score:.2f}")
                print()
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print()
    
    def demo_cosmos_db_specific_rewrites(self):
        """Demonstrate Cosmos DB specific query rewrites"""
        print("🎯 Cosmos DB Specific Query Rewrites")
        print("-" * 40)
        
        if not self.rewriter:
            print("❌ OpenAI API key not available. Running in demo mode.")
            return
        
        for i, query in enumerate(self.cosmos_db_queries, 1):
            print(f"\n--- Query {i} ---")
            print("Original:")
            print(query.strip())
            print()
            
            try:
                result = self.rewriter.rewrite_sql_query(query.strip(), optimization_type="cosmos_db")
                
                print("Optimized for Cosmos DB:")
                print(result.rewritten_query)
                print(f"Explanation: {result.explanation[:200]}...")
                print(f"Confidence: {result.confidence_score:.2f}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def demo_batch_processing(self):
        """Demonstrate batch processing of multiple queries"""
        print("\n📦 Batch Query Processing Demo")
        print("-" * 40)
        
        if not self.rewriter:
            print("❌ OpenAI API key not available. Running in demo mode.")
            return
        
        # Use a subset of queries for batch processing
        batch_queries = self.cosmos_db_queries[:3]
        
        print(f"Processing {len(batch_queries)} queries in batch...")
        
        try:
            results = self.rewriter.batch_rewrite_queries(batch_queries, optimization_type="performance")
            
            for i, result in enumerate(results, 1):
                print(f"\n--- Batch Result {i} ---")
                print(f"Original: {result.original_query[:100]}...")
                print(f"Rewritten: {result.rewritten_query[:100]}...")
                print(f"Confidence: {result.confidence_score:.2f}")
                
        except Exception as e:
            print(f"❌ Batch processing error: {str(e)}")
    
    def demo_query_analysis(self):
        """Demonstrate query pattern analysis"""
        print("\n📊 Query Pattern Analysis Demo")
        print("-" * 40)
        
        if not self.rewriter:
            print("❌ OpenAI API key not available. Running in demo mode.")
            return
        
        try:
            analysis = self.rewriter.analyze_query_patterns(self.cosmos_db_queries)
            
            print(f"Total queries analyzed: {analysis['total_queries']}")
            print(f"Common patterns found:")
            for pattern, count in analysis['common_patterns'].items():
                print(f"  • {pattern}: {count} queries")
            
        except Exception as e:
            print(f"❌ Analysis error: {str(e)}")
    
    def demo_similarity_search(self):
        """Demonstrate query similarity search"""
        print("\n🔍 Query Similarity Search Demo")
        print("-" * 40)
        
        if not self.rewriter:
            print("❌ OpenAI API key not available. Running in demo mode.")
            return
        
        try:
            # Create embeddings for queries
            print("Creating query embeddings...")
            vectorstore = self.rewriter.create_query_embeddings(self.cosmos_db_queries)
            
            # Test similarity search
            test_query = "SELECT users.name FROM users WHERE users.active = true"
            print(f"Finding similar queries to: {test_query}")
            
            similar_queries = self.rewriter.find_similar_queries(test_query, vectorstore, k=3)
            
            print(f"Found {len(similar_queries)} similar queries:")
            for i, doc in enumerate(similar_queries, 1):
                print(f"  {i}. {doc.page_content[:100]}...")
            
        except Exception as e:
            print(f"❌ Similarity search error: {str(e)}")
    
    def show_integration_examples(self):
        """Show integration examples with Cosmos DB"""
        print("\n🔗 Integration Examples")
        print("-" * 40)
        
        print("1. **Cosmos DB Container Integration:**")
        print("   ```python")
        print("   # Store rewritten queries back to Cosmos DB")
        print("   rewritten_queries = rewriter.batch_rewrite_queries(queries)")
        print("   for result in rewritten_queries:")
        print("       container.create_item({")
        print("           'id': generate_id(),")
        print("           'original_query': result.original_query,")
        print("           'rewritten_query': result.rewritten_query,")
        print("           'optimization_type': result.optimization_type,")
        print("           'confidence_score': result.confidence_score,")
        print("           'timestamp': datetime.utcnow().isoformat()")
        print("       })")
        print("   ```")
        print()
        
        print("2. **Query Performance Monitoring:**")
        print("   ```python")
        print("   # Track query performance before/after rewrite")
        print("   original_performance = measure_query_performance(original_query)")
        print("   rewritten_performance = measure_query_performance(rewritten_query)")
        print("   improvement = (original_performance - rewritten_performance) / original_performance")
        print("   ```")
        print()
        
        print("3. **Automated Query Optimization Pipeline:**")
        print("   ```python")
        print("   # Automated pipeline for query optimization")
        print("   def optimize_query_pipeline(query):")
        print("       # Analyze query patterns")
        print("       analysis = rewriter.analyze_query_patterns([query])")
        print("       ")
        print("       # Find similar queries for context")
        print("       similar = rewriter.find_similar_queries(query, vectorstore)")
        print("       ")
        print("       # Rewrite with context")
        print("       result = rewriter.rewrite_with_similar_context(query, vectorstore)")
        print("       ")
        print("       return result")
        print("   ```")
    
    def show_best_practices(self):
        """Show best practices for SQL prompt rewriting"""
        print("\n📚 Best Practices for SQL Prompt Rewriting")
        print("-" * 50)
        
        print("🎯 **Query Optimization Strategies:**")
        print("  • Use appropriate indexes for WHERE clauses")
        print("  • Minimize cross-partition queries in Cosmos DB")
        print("  • Optimize JOIN operations")
        print("  • Use efficient aggregation functions")
        print("  • Avoid SELECT * when possible")
        print()
        
        print("🔧 **Cosmos DB Specific Tips:**")
        print("  • Design queries around partition keys")
        print("  • Use point reads for single document retrieval")
        print("  • Optimize RU consumption")
        print("  • Use appropriate consistency levels")
        print("  • Leverage JSON path expressions")
        print()
        
        print("📊 **Performance Monitoring:**")
        print("  • Track query execution times")
        print("  • Monitor RU consumption")
        print("  • Analyze query patterns")
        print("  • Test with realistic data volumes")
        print("  • Validate query results")
        print()
        
        print("🔄 **Iterative Improvement:**")
        print("  • Start with general optimizations")
        print("  • Apply database-specific improvements")
        print("  • Test performance gains")
        print("  • Refine based on results")
        print("  • Document optimization decisions")
    
    def run_full_demo(self):
        """Run the complete demo"""
        print("🚀 Starting Cosmos DB SQL Prompt Rewrite Demo")
        print("=" * 60)
        
        self.show_cosmos_db_sql_rewrite_overview()
        
        if self.rewriter:
            self.demo_basic_rewrite()
            self.demo_cosmos_db_specific_rewrites()
            self.demo_batch_processing()
            self.demo_query_analysis()
            self.demo_similarity_search()
        else:
            print("\n❌ Demo Mode - OpenAI API key not available")
            print("To run full demo, set OPENAI_API_KEY in your .env file")
        
        self.show_integration_examples()
        self.show_best_practices()
        
        print("\n✨ Demo completed!")

def main():
    """Main function to run the demo"""
    demo = CosmosDBSQLRewriteDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main() 