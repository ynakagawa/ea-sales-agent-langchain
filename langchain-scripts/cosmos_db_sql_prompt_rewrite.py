#!/usr/bin/env python3
"""
Cosmos DB SQL Prompt Rewrite System using LangChain
This script provides functionality to rewrite and optimize SQL queries stored in Cosmos DB
"""

import os
import warnings
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS  # Temporarily disabled due to installation issues
from langchain.chains import RetrievalQA
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import re

# Suppress Cosmos DB warnings
warnings.filterwarnings("ignore", message="You appear to be connected to a CosmosDB cluster")

class SQLRewriteResult(BaseModel):
    """Result of SQL query rewrite operation"""
    original_query: str = Field(description="The original SQL query")
    rewritten_query: str = Field(description="The rewritten/optimized SQL query")
    optimization_type: str = Field(description="Type of optimization applied")
    explanation: str = Field(description="Explanation of changes made")
    performance_improvement: str = Field(description="Expected performance improvement")
    confidence_score: float = Field(description="Confidence score of the rewrite (0-1)")

class CosmosDBSQLPromptRewriter:
    """SQL Prompt Rewrite System for Cosmos DB using LangChain"""
    
    def __init__(self, openai_api_key: str, cosmos_endpoint: str = None, cosmos_key: str = None):
        self.openai_api_key = openai_api_key
        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_key = cosmos_key
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.output_parser = PydanticOutputParser(pydantic_object=SQLRewriteResult)
        
        # Initialize prompt templates
        self._initialize_prompts()
        
    def _initialize_prompts(self):
        """Initialize various prompt templates for SQL rewriting"""
        
        # Basic SQL rewrite prompt
        self.basic_rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL optimizer and query rewriter. Your task is to rewrite SQL queries to improve performance, readability, and maintainability while preserving the original logic.

Key optimization techniques:
1. Query structure optimization
2. Index usage optimization
3. Join optimization
4. Subquery optimization
5. WHERE clause optimization
6. SELECT clause optimization

IMPORTANT: Always provide your rewritten query in a SQL code block like this:
```sql
YOUR_REWRITTEN_QUERY_HERE
```

Then provide a detailed explanation of your changes and expected performance improvements."""),
            ("human", """Rewrite the following SQL query for better performance and readability:

Original Query:
{original_query}

Context: {context}

Please provide your rewritten query in a SQL code block, followed by a detailed explanation of your changes and expected performance improvements.""")
        ])
        
        # Performance optimization prompt
        self.performance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database performance expert specializing in query optimization. Focus on:
1. Reducing execution time
2. Minimizing resource usage
3. Optimizing for specific database engines
4. Using appropriate indexes
5. Avoiding common performance pitfalls

IMPORTANT: Always provide your optimized query in a SQL code block like this:
```sql
YOUR_OPTIMIZED_QUERY_HERE
```

Then provide specific optimization recommendations and expected performance improvements."""),
            ("human", """Optimize this SQL query for maximum performance:

Query: {query}

Database Type: {db_type}
Table Schema: {schema}

Please provide your optimized query in a SQL code block, followed by specific optimization recommendations and expected performance improvements.""")
        ])
        
        # Readability improvement prompt
        self.readability_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL code reviewer focused on improving query readability and maintainability. Focus on:
1. Clear naming conventions
2. Logical query structure
3. Proper formatting and indentation
4. Meaningful aliases
5. Simplified complex expressions

IMPORTANT: Always provide your improved query in a SQL code block like this:
```sql
YOUR_IMPROVED_QUERY_HERE
```

Then explain the readability improvements made."""),
            ("human", """Improve the readability of this SQL query:

Query: {query}

Please provide your improved query in a SQL code block, followed by an explanation of the readability improvements made.""")
        ])
        
        # Cosmos DB specific optimization prompt
        self.cosmos_db_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Cosmos DB expert. When optimizing queries for Cosmos DB, consider:
1. Partition key optimization
2. RU (Request Unit) consumption
3. Cross-partition queries
4. Indexing strategies
5. Query patterns for NoSQL
6. JSON path expressions
7. Array operations

IMPORTANT: Always provide your optimized query in a SQL code block like this:
```sql
YOUR_OPTIMIZED_QUERY_HERE
```

Then provide Cosmos DB specific optimization recommendations."""),
            ("human", """Optimize this query specifically for Cosmos DB:

Query: {query}
Container: {container}
Partition Key: {partition_key}

Please provide your optimized query in a SQL code block, followed by Cosmos DB specific optimization recommendations.""")
        ])
        
    def rewrite_sql_query(self, original_query: str, context: str = "", optimization_type: str = "general") -> SQLRewriteResult:
        """Rewrite a SQL query for better performance and readability"""
        
        try:
            if optimization_type == "performance":
                chain = LLMChain(llm=self.llm, prompt=self.performance_prompt)
                response = chain.run({
                    "query": original_query,
                    "db_type": "Cosmos DB",
                    "schema": context
                })
            elif optimization_type == "readability":
                chain = LLMChain(llm=self.llm, prompt=self.readability_prompt)
                response = chain.run({"query": original_query})
            elif optimization_type == "cosmos_db":
                chain = LLMChain(llm=self.llm, prompt=self.cosmos_db_prompt)
                response = chain.run({
                    "query": original_query,
                    "container": "your-container",
                    "partition_key": "your-partition-key"
                })
            else:
                chain = LLMChain(llm=self.llm, prompt=self.basic_rewrite_prompt)
                response = chain.run({
                    "original_query": original_query,
                    "context": context
                })
            
            # Parse the response to extract components
            return self._parse_rewrite_response(original_query, response, optimization_type)
            
        except Exception as e:
            return SQLRewriteResult(
                original_query=original_query,
                rewritten_query=original_query,
                optimization_type=optimization_type,
                explanation=f"Error during rewrite: {str(e)}",
                performance_improvement="None",
                confidence_score=0.0
            )
    
    def _parse_rewrite_response(self, original_query: str, response: str, optimization_type: str) -> SQLRewriteResult:
        """Parse the LLM response into a structured SQLRewriteResult"""
        
        try:
            # Try multiple patterns to extract SQL query from response
            rewritten_query = original_query  # Default to original
            explanation = response
            
            # Pattern 1: Look for SQL code blocks
            sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
            if sql_match:
                rewritten_query = sql_match.group(1).strip()
                explanation = response.replace(sql_match.group(0), "").strip()
            else:
                # Pattern 2: Look for SELECT statements
                sql_match = re.search(r'(SELECT\s+.*?)(?=\n\n|\n[A-Z]|$)', response, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    rewritten_query = sql_match.group(1).strip()
                    explanation = response.replace(sql_match.group(1), "").strip()
                else:
                    # Pattern 3: Look for any SQL-like statement
                    sql_match = re.search(r'(SELECT|UPDATE|DELETE|INSERT)\s+.*?(?=\n\n|\n[A-Z]|$)', response, re.DOTALL | re.IGNORECASE)
                    if sql_match:
                        rewritten_query = sql_match.group(0).strip()
                        explanation = response.replace(sql_match.group(0), "").strip()
            
            # Clean up explanation
            explanation = re.sub(r'^\s*[-*]\s*', '', explanation, flags=re.MULTILINE)
            explanation = explanation.strip()
            
            # Estimate confidence score based on response quality and changes
            confidence_score = 0.5  # Base confidence
            if rewritten_query != original_query:
                confidence_score += 0.3  # Bonus for actual changes
            if len(explanation) > 50:
                confidence_score += 0.2  # Bonus for detailed explanation
            confidence_score = min(1.0, confidence_score)
            
            return SQLRewriteResult(
                original_query=original_query,
                rewritten_query=rewritten_query,
                optimization_type=optimization_type,
                explanation=explanation,
                performance_improvement="Improved query structure and readability",
                confidence_score=confidence_score
            )
            
        except Exception as e:
            return SQLRewriteResult(
                original_query=original_query,
                rewritten_query=original_query,
                optimization_type=optimization_type,
                explanation=f"Error parsing response: {str(e)}",
                performance_improvement="None",
                confidence_score=0.0
            )
    
    def batch_rewrite_queries(self, queries: List[str], optimization_type: str = "general") -> List[SQLRewriteResult]:
        """Rewrite multiple SQL queries in batch"""
        results = []
        for query in queries:
            result = self.rewrite_sql_query(query, optimization_type=optimization_type)
            results.append(result)
        return results
    
    def create_query_embeddings(self, queries: List[str]):
        """Create embeddings for SQL queries for similarity search"""
        print("⚠️  FAISS not available - similarity search disabled")
        return None
    
    def find_similar_queries(self, query: str, vectorstore, k: int = 5) -> List[Document]:
        """Find similar SQL queries using embeddings"""
        print("⚠️  FAISS not available - returning empty list")
        return []
    
    def rewrite_with_similar_context(self, query: str, vectorstore, optimization_type: str = "general") -> SQLRewriteResult:
        """Rewrite a query using context from similar queries"""
        print("⚠️  FAISS not available - rewriting without context")
        return self.rewrite_sql_query(query, "", optimization_type)
    
    def analyze_query_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze patterns in SQL queries for optimization insights"""
        analysis = {
            "total_queries": len(queries),
            "common_patterns": {},
            "optimization_opportunities": [],
            "performance_issues": []
        }
        
        # Analyze common patterns
        for query in queries:
            query_upper = query.upper()
            
            # Count common clauses
            if "WHERE" in query_upper:
                analysis["common_patterns"]["where_clauses"] = analysis["common_patterns"].get("where_clauses", 0) + 1
            if "JOIN" in query_upper:
                analysis["common_patterns"]["joins"] = analysis["common_patterns"].get("joins", 0) + 1
            if "GROUP BY" in query_upper:
                analysis["common_patterns"]["group_by"] = analysis["common_patterns"].get("group_by", 0) + 1
            if "ORDER BY" in query_upper:
                analysis["common_patterns"]["order_by"] = analysis["common_patterns"].get("order_by", 0) + 1
            if "SUBQUERY" in query_upper or "(" in query:
                analysis["common_patterns"]["subqueries"] = analysis["common_patterns"].get("subqueries", 0) + 1
        
        return analysis

def demo_sql_rewrite_system():
    """Demonstrate the SQL rewrite system functionality"""
    
    print("🚀 Cosmos DB SQL Prompt Rewrite System Demo")
    print("=" * 50)
    
    # Sample SQL queries for demonstration
    sample_queries = [
        """
        SELECT u.name, u.email, p.title, p.content 
        FROM users u 
        INNER JOIN posts p ON u.id = p.user_id 
        WHERE u.created_at > '2023-01-01' 
        ORDER BY u.name
        """,
        
        """
        SELECT COUNT(*) as total_posts, 
               AVG(p.likes) as avg_likes,
               u.name
        FROM users u
        LEFT JOIN posts p ON u.id = p.user_id
        WHERE p.created_at BETWEEN '2023-01-01' AND '2023-12-31'
        GROUP BY u.id, u.name
        HAVING COUNT(*) > 5
        ORDER BY total_posts DESC
        """,
        
        """
        SELECT * FROM (
            SELECT u.name, COUNT(p.id) as post_count
            FROM users u
            LEFT JOIN posts p ON u.id = p.user_id
            GROUP BY u.id, u.name
        ) subquery
        WHERE post_count > 10
        """
    ]
    
    # Initialize the rewriter (using demo mode)
    print("📝 Initializing SQL Rewrite System...")
    rewriter = CosmosDBSQLPromptRewriter(
        openai_api_key="your-openai-api-key-here"
    )
    
    print("\n🔄 Testing SQL Query Rewrites...")
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Original: {query.strip()}")
        
        # Test different optimization types
        optimization_types = ["general", "performance", "readability"]
        
        for opt_type in optimization_types:
            print(f"\n🔧 {opt_type.title()} Optimization:")
            result = rewriter.rewrite_sql_query(query.strip(), optimization_type=opt_type)
            
            print(f"Rewritten: {result.rewritten_query}")
            print(f"Explanation: {result.explanation[:200]}...")
            print(f"Confidence: {result.confidence_score:.2f}")
    
    print("\n📊 Query Pattern Analysis...")
    analysis = rewriter.analyze_query_patterns(sample_queries)
    print(f"Total queries analyzed: {analysis['total_queries']}")
    print(f"Common patterns: {analysis['common_patterns']}")
    
    print("\n🎯 Creating Query Embeddings...")
    try:
        vectorstore = rewriter.create_query_embeddings(sample_queries)
        print("✅ Vector store created successfully")
        
        # Test similarity search
        test_query = "SELECT users.name FROM users WHERE users.active = true"
        similar = rewriter.find_similar_queries(test_query, vectorstore, k=2)
        print(f"Found {len(similar)} similar queries")
        
    except Exception as e:
        print(f"❌ Error creating embeddings: {str(e)}")
    
    print("\n✨ Demo completed!")

def main():
    """Main function to run the SQL rewrite system"""
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Running in demo mode...")
        demo_sql_rewrite_system()
        return
    
    # Initialize the rewriter
    rewriter = CosmosDBSQLPromptRewriter(openai_api_key=openai_api_key)
    
    # Example usage
    sample_query = """
    SELECT u.name, u.email, COUNT(p.id) as post_count
    FROM users u
    LEFT JOIN posts p ON u.id = p.user_id
    WHERE u.created_at > '2023-01-01'
    GROUP BY u.id, u.name, u.email
    HAVING COUNT(p.id) > 5
    ORDER BY post_count DESC
    """
    
    print("🔄 Rewriting SQL query...")
    result = rewriter.rewrite_sql_query(sample_query, optimization_type="performance")
    
    print(f"Original Query: {result.original_query}")
    print(f"Rewritten Query: {result.rewritten_query}")
    print(f"Optimization Type: {result.optimization_type}")
    print(f"Explanation: {result.explanation}")
    print(f"Performance Improvement: {result.performance_improvement}")
    print(f"Confidence Score: {result.confidence_score}")

if __name__ == "__main__":
    main() 