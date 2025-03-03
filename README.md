# LangChain RAG, Chains, Chat Models & Agents Project

## Overview

This project demonstrates advanced applications of **LangChain** using various components such as **Chat Models, Retrieval-Augmented Generation (RAG), Chains, Prompt Templates, and Agents**. The implementation includes **LLM-based conversations, AI-powered reasoning, document-based retrieval, and autonomous agent execution**.

The goal of this project is to showcase how **LLMs (Large Language Models)** can be integrated into different **AI-powered workflows**, providing enhanced natural language understanding, information retrieval, and decision-making capabilities.

---

## Project Structure
```commandline
LANGCHAIN
│── 1_chat_models/
│   ├── 1_chat_models_starter.py
│   ├── 2_models_conversation.py
│   ├── 3_chat_models_alt_models.py
│   ├── 4_chat_model_convo_with_user.py
│   ├── 5_chat_model_save_msgs_history.py
│── 2_prompt_templates/
│   ├── 2_prompt_templates_starter.py
│── 3_chains/
│   ├── 1_chains_basics.py
│   ├── 2_chains_inner_workings.py
│   ├── 3_1_chains_sequential.py
│   ├── 3_2_chains_sequential.py
│   ├── 4_chains_parallel.py
│   ├── 5_chains_conditional.py
│── 4_RAGs/
│   ├── db/
│   ├── documents/
│   ├── 1_basic_1.py
│   ├── 1_basic_2.py
│   ├── 2a_rag_metadata.py
│   ├── 2b_rag_metadata.py
│   ├── 3_rag_one_off_question.py
│── 5_agents/
│   ├── agents.py
│── .env
│── .gitignore
│── README.md
│── requirements.txt
```
---

## **1. Chat Models**

### **Description**
The chat models implemented in this project allow **interactive communication** with AI models using **LangChain** and **Groq LLMs**. They can handle **single-turn** or **multi-turn conversations**, store history, and simulate human-like responses.

### **Features**
- **Basic LLM Interaction** → Direct communication with the AI model.
- **Multi-Turn Conversations** → Chatbot maintains conversation history.
- **Alternative Model Comparisons** → Comparing different LLM models.
- **Conversation Logging** → Stores chat history in a **database (Firestore)**.

### **Use Cases**
- Building **AI chatbots** for customer support, education, or entertainment.
- Logging chat sessions for **AI model improvement** and **analysis**.
- Comparing responses from different **LLMs** to optimize results.

---

## **2. Prompt Templates**

### **Description**
Prompt templates **structure the input** provided to LLMs, ensuring consistency and clarity. **Well-designed prompts** lead to **better AI responses**.

### **Features**
- **Dynamic Prompt Generation** → Constructs AI prompts based on user input.
- **Structured Message Formatting** → Uses **System, Human, AI messages**.
- **Specialized Templates** → Emails, jokes, storytelling, job applications.

### **Use Cases**
- Creating **personalized AI-generated content** (emails, articles, summaries).
- Automating **marketing content generation** (advertisements, social media posts).
- Ensuring structured interactions with AI, improving **accuracy & coherence**.

---

## **3. Chains (LLM Pipelines)**

### **Description**
Chains allow **multiple LLM-powered tasks** to be executed **sequentially or in parallel**. They are essential for **multi-step AI workflows**.

### **Types of Chains Implemented**
1. **Basic Chains** → Single input → output processing.
2. **Sequential Chains** → Outputs from one step feed into the next.
3. **Parallel Chains** → Multiple independent computations running together.
4. **Conditional Chains** → AI dynamically selects different paths based on input.
5. **Interactive Chains** → AI transforms responses and translates them.

### **Features**
- **Sequential Processing** → AI completes one task before moving to the next.
- **Parallel Execution** → Multiple LLM tasks execute simultaneously.
- **Dynamic Decision Making** → AI chooses the best process based on input.

### **Use Cases**
- **Multi-step AI pipelines** → Summarization → Translation → Analysis.
- **Automated AI workflows** → Fact-checking, content enhancement.
- **AI-driven decision systems** → Dynamically selecting AI responses.

---

## **4. Retrieval-Augmented Generation (RAG)**

### **Description**
RAG enhances **LLM capabilities** by integrating **external knowledge retrieval**. Instead of relying only on pre-trained data, AI can fetch relevant **documents, articles, and books**.

### **How RAG Works**
1. **Document Processing** → AI ingests text files & splits them into smaller chunks.
2. **Vector Embeddings** → Converts text into **numerical vectors**.
3. **ChromaDB Storage** → Stores vectors for fast **semantic search**.
4. **Query Processing** → AI retrieves the most **relevant content**.
5. **LLM Integration** → AI generates responses based on retrieved data.

### **Features**
- **Document Chunking & Embeddings** → Efficient text retrieval.
- **Vector Search** → AI searches **similar content** efficiently.
- **Context-Aware AI Responses** → Responses are factually backed by documents.

### **Use Cases**
- **AI-powered knowledge bases** → Providing accurate answers from stored books.
- **Legal, Medical, & Research AI Assistants** → Searching for relevant cases/reports.
- **Enhancing AI Chatbots** → Reducing hallucinations by **grounding AI in real data**.

---

## **5. Autonomous AI Agents**

### **Description**
Agents enable **AI to make decisions** by **analyzing queries**, using **tools**, and **executing actions** autonomously.

### **Components of an Agent**
- **LLM Core** → The thinking unit of the agent.
- **Prompting Strategy** → Guides how the agent thinks & acts.
- **Toolset** → External functions AI can use (e.g., fetching system time).
- **Decision Process** → AI decides **when & how** to use tools.

### **How It Works**
1. **User Query** → "What is the current time in London? You are in Germany."
2. **AI Reasoning** → AI determines the **best action** (fetch time).
3. **Tool Execution** → Calls a function to get system time.
4. **Final Answer** → AI responds with **accurate results**.

### **Features**
- **ReAct Reasoning** → AI **thinks before acting**.
- **Autonomous Execution** → AI **uses tools** for accurate responses.
- **Debugging & Logging** → Monitors agent decision-making.

### **Use Cases**
- **Automated AI assistants** → Answering real-world queries accurately.
- **Decision-making bots** → Choosing the best tools dynamically.
- **AI-based workflow automation** → Streamlining repetitive AI tasks.

---

## **6. Installation & Setup**

### **Dependencies**
To install all required dependencies, run:
```commandline
pip install -r requirements.txt
```

### **Environment Variables**
Store API keys and configurations in a `.env` file.

### **Running the Project**
1. Clone the repository:
```commandline
git clone https://github.com/imaduddin27/Langchain
```


2. Run a specific module:
```commandline
python 1_chat_models/1_chat_models_starter.py
```

## Final Thoughts  

This project showcases the full potential of LangChain, covering:  

- AI-driven conversations  
- Smart prompt engineering  
- Advanced AI workflows with chains  
- Knowledge retrieval using RAG  
- Autonomous decision-making with AI agents  

With seamless integration of ChromaDB, Groq LLMs, and Firestore, this system is built for efficiency, scalability, and real-world applications.  




