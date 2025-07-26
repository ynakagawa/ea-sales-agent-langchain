#!/usr/bin/env python3
"""
Simple test script for SQL to Natural Language conversion
"""

from sql_to_natural_language import SQLToNaturalLanguage

def test_simple_conversion():
    """Test simple SQL to natural language conversion"""
    
    # Initialize the converter
    converter = SQLToNaturalLanguage()
    
    # Simple test query
    sql_query = """
    SELECT 
        customer_name,
        order_date,
        total_amount
    FROM orders
    WHERE order_date >= '2023-01-01'
    ORDER BY total_amount DESC
    LIMIT 10
    """
    
    print("🧪 Testing SQL to Natural Language Conversion")
    print("=" * 50)
    print(f"SQL Query:\n{sql_query.strip()}")
    
    # Convert to natural language
    natural_language = converter.convert(sql_query.strip())
    
    print(f"\n🔤 Natural Language Description:")
    print(natural_language)
    print("=" * 50)

def test_complex_conversion():
    """Test more complex SQL with joins and aggregates"""
    
    converter = SQLToNaturalLanguage()
    
    complex_sql = """
    SELECT 
        u.name as user_name,
        COUNT(o.id) as total_orders,
        SUM(o.total_amount) as total_spent,
        AVG(o.total_amount) as avg_order_value
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id
    WHERE u.created_at >= '2023-01-01'
    AND o.status = 'completed'
    GROUP BY u.id, u.name
    HAVING COUNT(o.id) >= 3
    ORDER BY total_spent DESC
    LIMIT 20
    """
    
    print("\n🧪 Testing Complex SQL Conversion")
    print("=" * 50)
    print(f"Complex SQL Query:\n{complex_sql.strip()}")
    
    natural_language = converter.convert(complex_sql.strip())
    
    print(f"\n🔤 Natural Language Description:")
    print(natural_language)
    print("=" * 50)

if __name__ == "__main__":
    test_simple_conversion()
    test_complex_conversion() 