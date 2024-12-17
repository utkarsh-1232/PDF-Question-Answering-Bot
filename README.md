# PDF Question Answering Bot
## Objective
Create a chatbot that can answer questions based on the content of a provided PDF document. If the chatbot cannot find the answer in the PDF, it should respond:

*“Sorry, I didn’t understand your question. Do you want to connect with a live agent?”*

## Requirements
1.	**PDF Understanding**:
- The chatbot should load the provided PDF and use its content as the knowledge base.
- It must accurately retrieve and display answers from the PDF content.

2.	**Fallback Response**:
- If the chatbot cannot find the answer in the PDF, it must provide a fallback response:
*“Sorry, I didn’t understand your question. Do you want to connect with a live agent?”*

3.	**User Interaction**:
-	Develop an interface (text-based or graphical) where users can ask questions and receive responses.
-	Ensure the interaction is clear and intuitive.

## Core Components
- embedding_model: `sentence-transformers/all-MiniLM-l6-v2`
- repo_id: `huggingfaceh4/zephyr-7b-alpha`
- prompt_template:
```
Answer the question using the context below as an AI Assistant. Be clear and concise. If the context does not contain enough information to answer the question, respond with:
"Sorry, I didn’t understand your question. Do you want to connect with a live agent?"

Context: {context}
Question: {question}
Answer:
```
- user interface: `streamlit`
