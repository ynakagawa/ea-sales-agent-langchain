#!/usr/bin/env python3
"""
Interactive SQL to Natural Language Converter
Run this script to interactively convert SQL queries to natural language
"""

from sql_to_natural_language import SQLToNaturalLanguage

def main():
    """Interactive SQL to natural language converter"""
    
    print("🔤 SQL to Natural Language Converter")
    print("=" * 50)
    print("Enter SQL queries and get natural language descriptions!")
    print("Type 'quit' to exit, 'help' for example queries")
    print("=" * 50)
    
    # Initialize the converter
    converter = SQLToNaturalLanguage()
    
    while True:
        print("\n📝 Enter your SQL query (or 'quit' to exit):")
        sql_input = input("SQL> ")
        
        if sql_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if sql_input.lower() == 'help':
            print("\n💡 Example SQL queries you can try:")
            print("""
1. Simple query:
   SELECT customer_name, order_date FROM orders WHERE order_date >= '2023-01-01'

2. With aggregation:
   SELECT category, COUNT(*) as count FROM products GROUP BY category

3. With joins:
   SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name

4. Complex query:
   SELECT p.name, AVG(r.rating) FROM products p INNER JOIN reviews r ON p.id = r.product_id WHERE p.price > 100 GROUP BY p.name HAVING AVG(r.rating) > 4 ORDER BY AVG(r.rating) DESC LIMIT 10
            """)
            continue
        
        if not sql_input.strip():
            print("❌ Please enter a valid SQL query")
            continue
        
        try:
            print("\n🔄 Converting SQL to natural language...")
            natural_language = converter.convert(sql_input.strip())
            
            print("\n🔤 Natural Language Description:")
            print("-" * 40)
            print(natural_language)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("💡 Make sure your SQL syntax is valid")

if __name__ == "__main__":
    main() 