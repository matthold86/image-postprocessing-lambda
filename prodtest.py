import json
from lambda_function import lambda_handler

# Load event data
with open('event.json', 'r') as file:
    event_data = json.load(file)

# Dummy context
class Context:
    function_name = 'test_lambda_function'
    memory_limit_in_mb = 128

# Invoke the lambda function
response = lambda_handler(event_data, Context())
print(response)
