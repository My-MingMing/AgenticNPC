import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from openai import OpenAI

client = OpenAI(
    base_url='your base url here',
    api_key='your api key here',
)

# Test claude-sonnet-4-6
print("Testing claude-sonnet-4-6...")
try:
    resp = client.chat.completions.create(
        model='claude-sonnet-4-6',
        messages=[{'role': 'user', 'content': 'Say "hello world" in a <test_cases> tag'}],
        max_tokens=100,
    )
    if hasattr(resp, 'choices'):
        print(f"OK: {resp.choices[0].message.content}")
    else:
        print(f"Unexpected type: {type(resp)}, value: {str(resp)[:200]}")
except Exception as e:
    print(f"Error: {e}")

# Test gpt-4o-mini
print("\nTesting gpt-4o-mini...")
try:
    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Say "hello world" in a <test_cases> tag'}],
        max_tokens=100,
    )
    if hasattr(resp, 'choices'):
        print(f"OK: {resp.choices[0].message.content}")
    else:
        print(f"Unexpected type: {type(resp)}, value: {str(resp)[:200]}")
except Exception as e:
    print(f"Error: {e}")
