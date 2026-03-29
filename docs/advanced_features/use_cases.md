# Chimera Advanced Features & Use Cases

## Overview

This guide explores advanced features and real-world use cases for Chimera, including guided generation, multi-model serving, RAG integration, and production deployment patterns.

## Table of Contents

- [Guided Generation](#guided-generation)
- [Multi-Model Serving](#multi-model-serving)
- [RAG Integration](#rag-integration)
- [Function Calling](#function-calling)
- [Vision-Language Models](#vision-language-models)
- [LoRA Adapters](#lora-adapters)
- [Speculative Decoding](#speculative-decoding)
- [Production Patterns](#production-patterns)
- [Enterprise Use Cases](#enterprise-use-cases)

---

## Guided Generation

Chimera supports structured output generation using various constraint methods.

### JSON Schema Guidance

Generate valid JSON output following a schema:

```python
from sglang import Engine, SamplingParams
import json

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

# Define JSON schema
schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Person's full name"
        },
        "age": {
            "type": "integer",
            "description": "Person's age in years"
        },
        "email": {
            "type": "string",
            "format": "email",
            "description": "Person's email address"
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of technical skills"
        }
    },
    "required": ["name", "age", "email", "skills"]
}

# Generate structured output
response = engine.generate(
    prompt="Generate a profile for a software engineer:",
    sampling_params=SamplingParams(
        guided_json=schema,
        max_new_tokens=300,
        temperature=0.7,
    ),
)

# Parse the result
profile = json.loads(response.text)
print(json.dumps(profile, indent=2))
```

**Output:**
```json
{
  "name": "Jane Smith",
  "age": 28,
  "email": "jane.smith@example.com",
  "skills": ["Python", "JavaScript", "Docker", "AWS"]
}
```

### Regex Guidance

Constrain output to match a regex pattern:

```python
from sglang import Engine, SamplingParams

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

# Generate phone number in specific format
response = engine.generate(
    prompt="Generate a US phone number:",
    sampling_params=SamplingParams(
        guided_regex=r"\(\d{3}\) \d{3}-\d{4}",
        max_new_tokens=20,
    ),
)

print(response.text)  # Output: (555) 123-4567
```

### Choice Guidance

Force model to choose from predefined options:

```python
from sglang import Engine, SamplingParams

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

# Sentiment analysis with fixed labels
response = engine.generate(
    prompt="Sentiment of 'I love this product!':",
    sampling_params=SamplingParams(
        guided_choice=["positive", "negative", "neutral"],
        max_new_tokens=10,
    ),
)

print(response.text)  # Output: positive
```

### Grammar Guidance (CFG)

Use context-free grammar for complex structures:

```python
from sglang import Engine, SamplingParams

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

# Define grammar for arithmetic expressions
grammar = r"""
?start: expression

?expression: term (("+" | "-") term)*
?term: factor (("*" | "/") factor)*
?factor: NUMBER
       | "(" expression ")"

NUMBER: /\d+/
"""

response = engine.generate(
    prompt="Generate a valid arithmetic expression:",
    sampling_params=SamplingParams(
        guided_grammar=grammar,
        max_new_tokens=50,
    ),
)

print(response.text)  # Output: (3 + 5) * 2
```

### Use Case: Form Extraction

```python
from sglang import Engine, SamplingParams
import json

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

# Extract structured data from text
text = """
John Doe applied for the position of Senior Software Engineer.
He has 8 years of experience and can be reached at john.doe@email.com.
His expected salary is $150,000 and he can start in 2 weeks.
"""

schema = {
    "type": "object",
    "properties": {
        "applicant_name": {"type": "string"},
        "position": {"type": "string"},
        "years_experience": {"type": "integer"},
        "email": {"type": "string"},
        "expected_salary": {"type": "integer"},
        "notice_period": {"type": "string"}
    },
    "required": ["applicant_name", "position", "email"]
}

response = engine.generate(
    prompt=f"Extract information from this text:\n\n{text}",
    sampling_params=SamplingParams(
        guided_json=schema,
        max_new_tokens=300,
    ),
)

data = json.loads(response.text)
print(json.dumps(data, indent=2))
```

---

## Multi-Model Serving

Run multiple models on the same infrastructure.

### Option 1: Multiple Server Instances

```bash
# Server 1: 8B model on port 30000
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.9

# Server 2: 70B model on port 30001 (multi-GPU)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --port 30001 \
  --tp-size 4 \
  --mem-fraction-static 0.85
```

### Option 2: Router Configuration

Use Chimera router for load balancing:

```bash
# Install router
pip install sglang-router

# Start router with multiple backends
chimera-router \
  --backends \
    "http://localhost:30000:model-8b" \
    "http://localhost:30001:model-70b" \
  --port 8080 \
  --routing-strategy "model-based"
```

**Client usage:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1")

# Route to specific model
response = client.chat.completions.create(
    model="model-8b",  # Routes to port 30000
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Option 3: Dynamic Model Loading

```python
from sglang import Runtime

class MultiModelService:
    def __init__(self):
        self.models = {
            "8b": Runtime("http://localhost:30000"),
            "70b": Runtime("http://localhost:30001"),
        }
    
    async def generate(self, model_size: str, prompt: str):
        runtime = self.models[model_size]
        return await runtime.generate(prompt)

# Usage
service = MultiModelService()
response = await service.generate("8b", "Quick question...")
```

---

## RAG Integration

Integrate Chimera with Retrieval-Augmented Generation pipelines.

### Basic RAG Pipeline

```python
from sglang import Engine, SamplingParams
from typing import List

class RAGService:
    def __init__(self, model_path: str, embedding_model: str = None):
        self.engine = Engine(model_path=model_path)
        # You can use sentence-transformers or similar for embeddings
    
    def retrieve(self, query: str, documents: List[str], top_k: int = 3) -> List[str]:
        """Simple retrieval based on keyword matching.
        Replace with vector search for production."""
        # In production, use FAISS, Pinecone, etc.
        return documents[:top_k]
    
    def generate(self, query: str, context: str) -> str:
        """Generate answer using retrieved context."""
        prompt = f"""
        Use the following context to answer the question.
        If the answer cannot be found in the context, say "I don't know."

        Context:
        {context}

        Question: {query}

        Answer:
        """
        
        response = self.engine.generate(
            prompt=prompt,
            sampling_params=SamplingParams(
                temperature=0.3,
                max_new_tokens=300,
            ),
        )
        
        return response.text

# Usage
rag = RAGService("meta-llama/Llama-3.1-8B-Instruct")

documents = [
    "Chimera is an LLM serving stack with CuteDSL kernels.",
    "CuteDSL provides optimized kernels for Hopper/Blackwell GPUs.",
    "FP8 quantization can improve throughput by 2-3x.",
]

query = "What is Chimera and what are its benefits?"
context = "\n".join(rag.retrieve(query, documents))
answer = rag.generate(query, context)

print(answer)
```

### Advanced RAG with Citations

```python
from sglang import Engine, SamplingParams
import json

class RAGWithCitations:
    def __init__(self, model_path: str):
        self.engine = Engine(model_path=model_path)
    
    def generate_with_citations(self, query: str, documents: dict) -> dict:
        """Generate answer with source citations."""
        
        # Format documents with IDs
        doc_text = "\n\n".join([
            f"[{i+1}] {content}" 
            for i, content in enumerate(documents.values())
        ])
        
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "source_id": {"type": "integer"}
                        },
                        "required": ["text", "source_id"]
                    }
                }
            },
            "required": ["answer", "citations"]
        }
        
        prompt = f"""
        Answer the question using the provided documents.
        Include citations for each claim using [1], [2], etc.

        Documents:
        {doc_text}

        Question: {query}
        """
        
        response = self.engine.generate(
            prompt=prompt,
            sampling_params=SamplingParams(
                guided_json=schema,
                temperature=0.3,
                max_new_tokens=500,
            ),
        )
        
        return json.loads(response.text)

# Usage
rag = RAGWithCitations("meta-llama/Llama-3.1-8B-Instruct")

documents = {
    "doc1": "Chimera uses CuteDSL for kernel optimization.",
    "doc2": "FP8 support is available on Hopper and Blackwell GPUs.",
    "doc3": "Tensor parallelism enables multi-GPU inference.",
}

result = rag.generate_with_citations(
    "How does Chimera achieve high performance?",
    documents
)

print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
```

---

## Function Calling

Implement function calling capabilities similar to OpenAI's function calling.

### Basic Function Calling

```python
from sglang import Engine, SamplingParams
import json

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

# Define available functions
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_time",
        "description": "Get current time for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name"
                }
            },
            "required": ["timezone"]
        }
    }
]

def execute_function(name: str, arguments: dict):
    """Execute the called function."""
    if name == "get_weather":
        return f"Weather in {arguments['location']}: Sunny, 25°C"
    elif name == "get_time":
        return f"Time in {arguments['timezone']}: 14:30"
    return "Function not found"

# Function calling prompt
prompt = """
You are a helpful assistant with access to these functions:

{functions}

User: What's the weather in Tokyo?
Assistant:
""".format(functions=json.dumps(functions, indent=2))

# Schema for function call response
function_schema = {
    "type": "object",
    "properties": {
        "function_name": {"type": "string"},
        "arguments": {"type": "object"}
    },
    "required": ["function_name", "arguments"]
}

response = engine.generate(
    prompt=prompt,
    sampling_params=SamplingParams(
        guided_json=function_schema,
        temperature=0,
        max_new_tokens=200,
    ),
)

# Parse and execute function call
call = json.loads(response.text)
result = execute_function(call["function_name"], call["arguments"])

print(f"Called: {call['function_name']}")
print(f"Arguments: {call['arguments']}")
print(f"Result: {result}")
```

### Multi-Turn Function Calling

```python
from sglang import Engine, SamplingParams
import json

class FunctionCallingAgent:
    def __init__(self, model_path: str):
        self.engine = Engine(model_path=model_path)
        self.functions = {
            "search": self.search,
            "calculate": self.calculate,
            "translate": self.translate,
        }
    
    def search(self, query: str) -> str:
        return f"Search results for '{query}': [result1, result2]"
    
    def calculate(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"Result: {result}"
        except:
            return "Invalid expression"
    
    def translate(self, text: str, target_lang: str) -> str:
        return f"Translation to {target_lang}: [translated text]"
    
    def chat(self, user_message: str, conversation_history: list = None) -> str:
        functions_desc = json.dumps([
            {
                "name": "search",
                "description": "Search for information",
                "parameters": {"query": "string"}
            },
            {
                "name": "calculate",
                "description": "Calculate mathematical expression",
                "parameters": {"expression": "string"}
            },
            {
                "name": "translate",
                "description": "Translate text",
                "parameters": {"text": "string", "target_lang": "string"}
            }
        ], indent=2)
        
        history = "\n".join(conversation_history) if conversation_history else ""
        
        prompt = f"""
        You are a helpful assistant with these capabilities:
        {functions_desc}

        Conversation history:
        {history}

        User: {user_message}
        
        If you need to call a function, respond with JSON:
        {{"action": "function_name", "parameters": {{...}}}}
        
        Otherwise, respond normally.
        
        Assistant:
        """
        
        response = self.engine.generate(
            prompt=prompt,
            sampling_params=SamplingParams(
                temperature=0.7,
                max_new_tokens=300,
            ),
        )
        
        # Try to parse as function call
        try:
            call = json.loads(response.text.strip())
            if "action" in call and "parameters" in call:
                func = self.functions.get(call["action"])
                if func:
                    result = func(**call["parameters"])
                    return f"Function result: {result}"
        except:
            pass
        
        return response.text

# Usage
agent = FunctionCallingAgent("meta-llama/Llama-3.1-8B-Instruct")

messages = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    response = agent.chat(user_input, messages[-4:])  # Last 4 messages
    print(f"Assistant: {response}")
    messages.append(f"User: {user_input}")
    messages.append(f"Assistant: {response}")
```

---

## Vision-Language Models

Use multimodal models for image understanding.

### Image Captioning

```python
from sglang import Engine, SamplingParams
import requests
from PIL import Image
import io

engine = Engine(model_path="llava-hf/llava-v1.6-mistral-7b-hf")

# Load image
image_url = "https://example.com/image.jpg"
response = requests.get(image_url)
image = Image.open(io.BytesIO(response.content))

# Generate caption
prompt = f"<image>\nDescribe this image in detail:"

# Note: Image passing depends on implementation
response = engine.generate(
    prompt=prompt,
    image=image,  # Or image_path="path/to/image.jpg"
    sampling_params=SamplingParams(
        max_new_tokens=200,
        temperature=0.7,
    ),
)

print(response.text)
```

### Visual Q&A

```python
from sglang import Engine, SamplingParams

engine = Engine(model_path="llava-hf/llava-v1.6-mistral-7b-hf")

image_path = "document.png"
prompt = f"""
<image>
Answer the following questions about this document:

1. What is the title of this document?
2. What is the main topic?
3. List any dates mentioned.
"""

response = engine.generate(
    prompt=prompt,
    image_path=image_path,
    sampling_params=SamplingParams(
        max_new_tokens=500,
        temperature=0.3,
    ),
)

print(response.text)
```

---

## LoRA Adapters

Use LoRA adapters for task-specific fine-tuning.

### Loading LoRA Adapters

```python
from sglang import Engine, SamplingParams

# Load base model with LoRA
engine = Engine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    lora_path="path/to/lora/adapter",
    lora_scale=1.0,
)

# Generate with LoRA
response = engine.generate(
    prompt="Translate to French: Hello, how are you?",
    sampling_params=SamplingParams(max_new_tokens=50),
)

print(response.text)
```

### Multiple LoRA Adapters

```python
from sglang import Engine, SamplingParams

engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

# Load multiple adapters
adapters = {
    "translation": "path/to/translation/lora",
    "summarization": "path/to/summary/lora",
    "code": "path/to/code/lora",
}

for name, path in adapters.items():
    engine.load_lora(name, path)

# Use specific adapter
response = engine.generate(
    prompt="Summarize this article...",
    sampling_params=SamplingParams(max_new_tokens=200),
    lora_adapter="summarization",
)
```

---

## Speculative Decoding

Use speculative decoding for faster inference.

### Basic Speculative Decoding

```python
from sglang import Engine, SamplingParams

# Small draft model + large target model
draft_engine = Engine(model_path="TinyLlama/TinyLlama-1.1B-Chat")
target_engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

class SpeculativeDecoder:
    def __init__(self, draft, target, k: int = 4):
        self.draft = draft
        self.target = target
        self.k = k  # Number of draft tokens
    
    def generate(self, prompt: str) -> str:
        # Draft model generates k tokens
        draft_response = self.draft.generate(
            prompt=prompt,
            sampling_params=SamplingParams(max_new_tokens=self.k),
        )
        
        # Target model verifies and corrects
        final_prompt = f"{prompt} {draft_response.text}"
        final_response = self.target.generate(
            prompt=final_prompt,
            sampling_params=SamplingParams(max_new_tokens=self.k),
        )
        
        return final_response.text

decoder = SpeculativeDecoder(draft_engine, target_engine)
response = decoder.generate("The future of AI is")
print(response)
```

---

## Production Patterns

### Pattern 1: Request Queue with Priority

```python
from sglang import Engine, SamplingParams
import asyncio
from dataclasses import dataclass
from enum import Enum
import heapq

class Priority(Enum):
    HIGH = 1
    NORMAL = 2
    LOW = 3

@dataclass
class Request:
    id: str
    prompt: str
    priority: Priority
    timestamp: float
    
    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp

class PriorityQueue:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.queue = []
        self.processing = False
    
    async def submit(self, request: Request):
        heapq.heappush(self.queue, request)
        if not self.processing:
            asyncio.create_task(self.process())
    
    async def process(self):
        self.processing = True
        while self.queue:
            request = heapq.heappop(self.queue)
            response = self.engine.generate(
                prompt=request.prompt,
                sampling_params=SamplingParams(max_new_tokens=256),
            )
            # Handle response (save, send, etc.)
            print(f"Request {request.id}: {response.text[:100]}...")
        self.processing = False

# Usage
engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
queue = PriorityQueue(engine)

# Submit requests with different priorities
await queue.submit(Request("1", "High priority task", Priority.HIGH, time.time()))
await queue.submit(Request("2", "Normal task", Priority.NORMAL, time.time()))
await queue.submit(Request("3", "Low priority task", Priority.LOW, time.time()))
```

### Pattern 2: Rate Limiting

```python
from sglang import Engine, SamplingParams
from collections import deque
import time

class RateLimitedEngine:
    def __init__(self, engine: Engine, requests_per_second: int):
        self.engine = engine
        self.rps = requests_per_second
        self.timestamps = deque()
    
    def generate(self, prompt: str, **kwargs):
        # Wait if rate limit exceeded
        now = time.time()
        while len(self.timestamps) >= self.rps:
            oldest = self.timestamps[0]
            if now - oldest < 1.0:
                time.sleep(1.0 - (now - oldest))
                now = time.time()
            else:
                self.timestamps.popleft()
        
        self.timestamps.append(time.time())
        return self.engine.generate(prompt, **kwargs)

# Usage
engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
rate_limited = RateLimitedEngine(engine, requests_per_second=10)

# All requests are rate-limited to 10/second
for i in range(100):
    response = rate_limited.generate(f"Request {i}")
```

### Pattern 3: Circuit Breaker

```python
from sglang import Engine, SamplingParams
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, engine: Engine, failure_threshold: int = 5, timeout: int = 60):
        self.engine = engine
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def generate(self, prompt: str, **kwargs):
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            response = self.engine.generate(prompt, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return response
        except Exception as e:
            self.failures += 1
            self.last_failure_time = datetime.now()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise

# Usage
engine = Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
protected_engine = CircuitBreaker(engine)

try:
    response = protected_engine.generate("Test prompt")
except Exception as e:
    print(f"Circuit breaker triggered: {e}")
```

---

## Enterprise Use Cases

### Use Case 1: Customer Support Chatbot

```python
from sglang import Engine, SamplingParams
import json

class SupportChatbot:
    def __init__(self, model_path: str, knowledge_base: dict):
        self.engine = Engine(model_path=model_path)
        self.kb = knowledge_base
    
    def retrieve_kb(self, query: str) -> str:
        """Retrieve relevant KB articles."""
        # Simple keyword matching (use vector search in production)
        relevant = []
        for title, content in self.kb.items():
            if any(word in content.lower() for word in query.lower().split()):
                relevant.append(f"{title}: {content}")
        return "\n\n".join(relevant[:3])
    
    def respond(self, user_message: str, conversation_history: list) -> str:
        context = self.retrieve_kb(user_message)
        
        schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "escalate": {"type": "boolean"},
                "category": {"type": "string"}
            }
        }
        
        prompt = f"""
        You are a customer support assistant. Use the knowledge base to answer questions.
        If you cannot find the answer, set escalate to true.

        Knowledge Base:
        {context}

        Conversation:
        {'\n'.join(conversation_history[-4:])}

        Customer: {user_message}
        """
        
        response = self.engine.generate(
            prompt=prompt,
            sampling_params=SamplingParams(
                guided_json=schema,
                temperature=0.3,
                max_new_tokens=300,
            ),
        )
        
        return json.loads(response.text)

# Usage
kb = {
    "Returns": "Items can be returned within 30 days of purchase.",
    "Shipping": "Standard shipping takes 5-7 business days.",
    "Payment": "We accept credit cards, PayPal, and Apple Pay.",
}

chatbot = SupportChatbot("meta-llama/Llama-3.1-8B-Instruct", kb)
result = chatbot.respond("How do I return an item?", ["Customer: Hi"])
print(result)
```

### Use Case 2: Code Review Assistant

```python
from sglang import Engine, SamplingParams

class CodeReviewAssistant:
    def __init__(self, model_path: str):
        self.engine = Engine(model_path=model_path)
    
    def review(self, code: str, language: str = "python") -> dict:
        prompt = f"""
        Review the following {language} code for:
        1. Bugs and errors
        2. Performance issues
        3. Code style and best practices
        4. Security vulnerabilities

        Code:
        ```{language}
        {code}
        ```

        Provide your review in JSON format with these fields:
        - issues: list of {{severity, description, suggestion}}
        - overall_score: 1-10
        - summary: brief summary
        """
        
        schema = {
            "type": "object",
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity": {"type": "string"},
                            "description": {"type": "string"},
                            "suggestion": {"type": "string"}
                        }
                    }
                },
                "overall_score": {"type": "integer"},
                "summary": {"type": "string"}
            }
        }
        
        response = self.engine.generate(
            prompt=prompt,
            sampling_params=SamplingParams(
                guided_json=schema,
                temperature=0.2,
                max_new_tokens=800,
            ),
        )
        
        return json.loads(response.text)

# Usage
assistant = CodeReviewAssistant("meta-llama/Llama-3.1-8B-Instruct")

code = """
def get_user_data(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
"""

review = assistant.review(code)
print(json.dumps(review, indent=2))
```

### Use Case 3: Document Analysis Pipeline

```python
from sglang import Engine, SamplingParams
from typing import List

class DocumentAnalyzer:
    def __init__(self, model_path: str):
        self.engine = Engine(model_path=model_path)
    
    def analyze(self, document: str) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_points": {"type": "array", "items": {"type": "string"}},
                "entities": {
                    "type": "object",
                    "properties": {
                        "people": {"type": "array", "items": {"type": "string"}},
                        "organizations": {"type": "array", "items": {"type": "string"}},
                        "dates": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "sentiment": {"type": "string"},
                "category": {"type": "string"}
            }
        }
        
        prompt = f"""
        Analyze the following document and extract key information.

        Document:
        {document[:10000]}  # Truncate if too long

        Provide your analysis in JSON format.
        """
        
        response = self.engine.generate(
            prompt=prompt,
            sampling_params=SamplingParams(
                guided_json=schema,
                temperature=0.1,
                max_new_tokens=1000,
            ),
        )
        
        return json.loads(response.text)
    
    def batch_analyze(self, documents: List[str]) -> List[dict]:
        """Analyze multiple documents in batch."""
        results = []
        for doc in documents:
            result = self.analyze(doc)
            results.append(result)
        return results

# Usage
analyzer = DocumentAnalyzer("meta-llama/Llama-3.1-70B-Instruct")

document = """
[Long document text...]
"""

analysis = analyzer.analyze(document)
print(json.dumps(analysis, indent=2))
```

---

**Last Updated**: March 29, 2026
**Version**: Chimera v1.0
