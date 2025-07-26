#!/usr/bin/env python3
"""
SQL to Natural Language Rewriting System
Converts SQL queries to human-readable natural language descriptions
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
import json

@dataclass
class SQLComponents:
    """Data class to hold parsed SQL components"""
    select_clause: List[str]
    from_clause: List[str]
    where_clause: Optional[str]
    group_by: Optional[List[str]]
    order_by: Optional[List[str]]
    limit: Optional[int]
    join_clauses: List[Dict]
    aggregate_functions: List[str]
    table_aliases: Dict[str, str]

class SQLParser:
    """Parser to extract components from SQL queries"""
    
    def __init__(self):
        self.aggregate_patterns = [
            r'COUNT\([^)]*\)',
            r'SUM\([^)]*\)',
            r'AVG\([^)]*\)',
            r'MAX\([^)]*\)',
            r'MIN\([^)]*\)',
            r'DISTINCT\s+[^,\s]+'
        ]
    
    def parse_sql(self, sql_query: str) -> SQLComponents:
        """Parse SQL query and extract components"""
        # Normalize the query
        sql = sql_query.strip().upper()
        
        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        select_clause = []
        if select_match:
            select_fields = select_match.group(1).strip()
            select_clause = [field.strip() for field in select_fields.split(',')]
        
        # Extract FROM clause
        from_match = re.search(r'FROM\s+(.+?)(?:\s+WHERE|\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        from_clause = []
        if from_match:
            from_tables = from_match.group(1).strip()
            from_clause = [table.strip() for table in from_tables.split(',')]
        
        # Extract WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        where_clause = where_match.group(1).strip() if where_match else None
        
        # Extract GROUP BY clause
        group_match = re.search(r'GROUP\s+BY\s+(.+?)(?:\s+ORDER\s+BY|\s+LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        group_by = None
        if group_match:
            group_fields = group_match.group(1).strip()
            group_by = [field.strip() for field in group_fields.split(',')]
        
        # Extract ORDER BY clause
        order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:\s+LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        order_by = None
        if order_match:
            order_fields = order_match.group(1).strip()
            order_by = [field.strip() for field in order_fields.split(',')]
        
        # Extract LIMIT clause
        limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
        limit = int(limit_match.group(1)) if limit_match else None
        
        # Extract JOIN clauses
        join_clauses = self._extract_joins(sql)
        
        # Extract aggregate functions
        aggregate_functions = self._extract_aggregates(sql)
        
        # Extract table aliases
        table_aliases = self._extract_aliases(sql)
        
        return SQLComponents(
            select_clause=select_clause,
            from_clause=from_clause,
            where_clause=where_clause,
            group_by=group_by,
            order_by=order_by,
            limit=limit,
            join_clauses=join_clauses,
            aggregate_functions=aggregate_functions,
            table_aliases=table_aliases
        )
    
    def _extract_joins(self, sql: str) -> List[Dict]:
        """Extract JOIN clauses from SQL"""
        joins = []
        join_pattern = r'(INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN)\s+([^\s]+)\s+ON\s+(.+?)(?=\s+(?:INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|WHERE|GROUP\s+BY|ORDER\s+BY|LIMIT|$))'
        
        for match in re.finditer(join_pattern, sql, re.IGNORECASE | re.DOTALL):
            joins.append({
                'type': match.group(1).strip(),
                'table': match.group(2).strip(),
                'condition': match.group(3).strip()
            })
        
        return joins
    
    def _extract_aggregates(self, sql: str) -> List[str]:
        """Extract aggregate functions from SQL"""
        aggregates = []
        for pattern in self.aggregate_patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE)
            aggregates.extend(matches)
        return aggregates
    
    def _extract_aliases(self, sql: str) -> Dict[str, str]:
        """Extract table aliases from SQL"""
        aliases = {}
        alias_pattern = r'FROM\s+([^\s]+)\s+AS\s+([^\s,]+)'
        for match in re.finditer(alias_pattern, sql, re.IGNORECASE):
            aliases[match.group(2)] = match.group(1)
        return aliases

class NaturalLanguageOutputParser(BaseOutputParser):
    """Parser for natural language output"""
    
    def parse(self, text: str) -> str:
        """Parse the natural language output"""
        # Clean up the output
        text = text.strip()
        # Remove any markdown formatting if present
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        return text.strip()

class SQLToNaturalLanguage:
    """Main class for converting SQL to natural language"""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """Initialize the SQL to natural language converter"""
        self.parser = SQLParser()
        self.llm = ChatOpenAI(
            model_name=llm_model,
            temperature=temperature,
            openai_api_key='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'
        )
        self.output_parser = NaturalLanguageOutputParser()
        
        # Create the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["sql_query", "sql_components", "context"],
            template="""
You are an expert SQL analyst who converts SQL queries into clear, natural language descriptions.

SQL Query: {sql_query}

Parsed Components:
{sql_components}

Additional Context: {context}

Please provide a clear, natural language description of what this SQL query does. Focus on:
1. What data is being retrieved
2. From which tables/sources
3. Any filtering conditions
4. Any grouping or aggregation
5. Any sorting or limiting

Write in a business-friendly tone that a non-technical person can understand.

Natural Language Description:
"""
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            output_parser=self.output_parser
        )
    
    def _format_components(self, components: SQLComponents) -> str:
        """Format SQL components for the prompt"""
        formatted = []
        
        if components.select_clause:
            formatted.append(f"SELECT: {', '.join(components.select_clause)}")
        
        if components.from_clause:
            formatted.append(f"FROM: {', '.join(components.from_clause)}")
        
        if components.join_clauses:
            joins = []
            for join in components.join_clauses:
                joins.append(f"{join['type']} {join['table']} ON {join['condition']}")
            formatted.append(f"JOINS: {'; '.join(joins)}")
        
        if components.where_clause:
            formatted.append(f"WHERE: {components.where_clause}")
        
        if components.group_by:
            formatted.append(f"GROUP BY: {', '.join(components.group_by)}")
        
        if components.order_by:
            formatted.append(f"ORDER BY: {', '.join(components.order_by)}")
        
        if components.limit:
            formatted.append(f"LIMIT: {components.limit}")
        
        if components.aggregate_functions:
            formatted.append(f"AGGREGATES: {', '.join(components.aggregate_functions)}")
        
        if components.table_aliases:
            aliases = [f"{alias} -> {table}" for alias, table in components.table_aliases.items()]
            formatted.append(f"ALIASES: {', '.join(aliases)}")
        
        return "\n".join(formatted)
    
    def convert(self, sql_query: str, context: str = "") -> str:
        """Convert SQL query to natural language"""
        try:
            # Parse the SQL query
            components = self.parser.parse_sql(sql_query)
            formatted_components = self._format_components(components)
            
            # Generate natural language description
            result = self.chain.run({
                "sql_query": sql_query,
                "sql_components": formatted_components,
                "context": context or "No additional context provided."
            })
            
            return result
        
        except Exception as e:
            return f"Error converting SQL to natural language: {str(e)}"
    
    def convert_batch(self, sql_queries: List[str], context: str = "") -> List[str]:
        """Convert multiple SQL queries to natural language"""
        results = []
        for sql in sql_queries:
            result = self.convert(sql, context)
            results.append(result)
        return results

def main():
    """Example usage and testing"""
    # Initialize the converter
    converter = SQLToNaturalLanguage()
    
    # Example SQL queries
    example_queries = [
        """
        SELECT 
            u.name,
            COUNT(o.id) as order_count,
            SUM(o.total_amount) as total_spent
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at >= '2023-01-01'
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 5
        ORDER BY total_spent DESC
        LIMIT 10
        """,
        
        """
        SELECT 
            p.category,
            AVG(p.price) as avg_price,
            MAX(p.price) as max_price
        FROM products p
        WHERE p.in_stock = true
        GROUP BY p.category
        ORDER BY avg_price DESC
        """,
        
        """
        SELECT 
            c.name as customer_name,
            o.order_date,
            o.total_amount
        FROM customers c
        INNER JOIN orders o ON c.id = o.customer_id
        WHERE o.order_date BETWEEN '2023-01-01' AND '2023-12-31'
        AND o.status = 'completed'
        ORDER BY o.total_amount DESC
        LIMIT 50
        """
    ]
    
    print("🧪 Testing SQL to Natural Language Conversion")
    print("=" * 60)
    
    for i, sql in enumerate(example_queries, 1):
        print(f"\n📝 Example {i}:")
        print(f"SQL Query:\n{sql.strip()}")
        
        natural_language = converter.convert(sql.strip())
        print(f"\n🔤 Natural Language:\n{natural_language}")
        print("-" * 60)

if __name__ == "__main__":
    main() 