# SQL to Natural Language Conversion System

This system converts SQL queries into human-readable natural language descriptions using LangChain and OpenAI's language models.

## Features

- **SQL Parsing**: Extracts and analyzes SQL query components
- **Natural Language Generation**: Converts SQL to business-friendly descriptions
- **Batch Processing**: Handle multiple queries at once
- **Context Awareness**: Include additional context for better descriptions
- **Error Handling**: Graceful handling of parsing errors

## Installation

1. Ensure you have the required packages installed:
```bash
pip install langchain langchain-community langsmith openai
```

2. Set up your OpenAI API key in the script or as an environment variable.

## Usage

### Basic Usage

```python
from sql_to_natural_language import SQLToNaturalLanguage

# Initialize the converter
converter = SQLToNaturalLanguage()

# Convert a SQL query to natural language
sql_query = """
SELECT customer_name, order_date, total_amount
FROM orders
WHERE order_date >= '2023-01-01'
ORDER BY total_amount DESC
LIMIT 10
"""

natural_language = converter.convert(sql_query)
print(natural_language)
```

### Advanced Usage with Context

```python
# Add business context for better descriptions
context = "This query is for the monthly sales report dashboard"
natural_language = converter.convert(sql_query, context=context)
```

### Batch Processing

```python
sql_queries = [
    "SELECT * FROM users WHERE active = true",
    "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
    "SELECT product_name, SUM(quantity) FROM sales GROUP BY product_name"
]

results = converter.convert_batch(sql_queries)
for i, result in enumerate(results):
    print(f"Query {i+1}: {result}")
```

## Supported SQL Features

The system can handle:

- **SELECT** clauses with field selection
- **FROM** clauses with table names
- **WHERE** conditions and filtering
- **JOIN** operations (INNER, LEFT, RIGHT, FULL)
- **GROUP BY** aggregations
- **ORDER BY** sorting
- **LIMIT** restrictions
- **Aggregate functions** (COUNT, SUM, AVG, MAX, MIN)
- **Table aliases**
- **HAVING** clauses

## Example Output

**Input SQL:**
```sql
SELECT 
    u.name as user_name,
    COUNT(o.id) as total_orders,
    SUM(o.total_amount) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at >= '2023-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 5
ORDER BY total_spent DESC
LIMIT 10
```

**Output Natural Language:**
```
This query retrieves user information along with their order statistics. It selects the user's name, counts the total number of orders for each user, and calculates the sum of all order amounts. The data comes from the users table joined with the orders table, matching users to their orders. It filters for users created on or after January 1, 2023, groups the results by user ID and name, and only includes users who have more than 5 orders. The results are sorted by total spending in descending order and limited to the top 10 users.
```

## Configuration

### Model Settings

You can customize the LLM model and parameters:

```python
converter = SQLToNaturalLanguage(
    llm_model="gpt-4",  # Use GPT-4 for better results
    temperature=0.1     # Lower temperature for more consistent output
)
```

### Custom Prompts

The system uses a predefined prompt template, but you can modify it in the `SQLToNaturalLanguage` class:

```python
# Modify the prompt_template in the __init__ method
self.prompt_template = PromptTemplate(
    input_variables=["sql_query", "sql_components", "context"],
    template="Your custom prompt template here..."
)
```

## Testing

Run the test script to see examples:

```bash
python test_sql_converter.py
```

Or run the full example with multiple test cases:

```bash
python sql_to_natural_language.py
```

## Error Handling

The system includes error handling for:

- Invalid SQL syntax
- Missing required clauses
- API errors
- Parsing failures

Errors are returned as descriptive error messages rather than crashing the application.

## Performance Considerations

- **API Costs**: Each conversion uses OpenAI API calls
- **Rate Limiting**: Consider implementing rate limiting for batch processing
- **Caching**: Consider caching results for repeated queries
- **Model Selection**: GPT-4 provides better results but costs more than GPT-3.5-turbo

## Extending the System

### Adding New SQL Features

To support additional SQL features, modify the `SQLParser` class:

1. Add new regex patterns in the appropriate method
2. Update the `SQLComponents` dataclass if needed
3. Modify the `_format_components` method to include new components

### Custom Output Formats

Create a new output parser by extending `BaseOutputParser`:

```python
class CustomOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Your custom parsing logic
        return processed_text
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your OpenAI API key is valid and has sufficient credits
2. **Import Errors**: Make sure all required packages are installed
3. **Parsing Errors**: Check that your SQL syntax is valid
4. **Rate Limiting**: Implement delays between API calls if needed

### Debug Mode

Add debug logging to see the parsed SQL components:

```python
# In the convert method, add:
print(f"Parsed components: {formatted_components}")
```

## License

This project is open source and available under the MIT License. 