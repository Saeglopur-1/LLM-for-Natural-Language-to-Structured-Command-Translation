This project is to transfer natural languages to structured-command translation.

Here's the data flow for our system:

User Input: A natural language string (e.g., "Check the database health every hour").

Retriever (Fine-Tuned Sentence Transformer):

The user's input is embedded into a vector.

This vector is used to search a FAISS vector store.

The vector store contains embeddings of command templates or examples of similar commands.

The retriever fetches the most relevant command templates (e.g., a template for "scheduled recurring checks").

Generator (LLM with LangChain):

A prompt is constructed that includes:

The original user input.

The retrieved command templates/examples.

Instructions to parse the input and generate a structured JSON output based on a predefined schema.

The LLM processes this augmented prompt.

Structured Output (JSON): The LLM, guided by the retrieved context and the prompt, generates a validated JSON object.

