# Enterprise CRAG Engine (Agentic Document QA)

An Enterprise-Grade Corrective Retrieval-Augmented Generation (CRAG) microservice built with **LangGraph**, **LangChain**, and **pgVector**.

This project moves beyond standard linear RAG pipelines by implementing a self-reflecting, agentic DAG (Directed Acyclic Graph) workflow. The agent dynamically retrieves documents, grades their relevance, and self-corrects by rewriting search queries if the initial retrieval fails to yield relevant context.

## Core Architecture
- **Agentic Orchestration:** LangGraph DAG for routing, grading, and self-correction.
- **Vector Database:** PostgreSQL with the `pgVector` extension for highly scalable, relational semantic search.
- **Embeddings & Inference:** OpenAI `text-embedding-3-small` and `gpt-4o-mini`.
- **Backend API:** Cloud-Native Flask REST API with OWASP security best practices (Rate Limiting, secure headers, API Key auth).
- **Observability:** Integrated with LangSmith for full LLMOps tracing and token monitoring.

## Features
1. **Semantic Document Ingestion:** Intelligent chunking with overlapping windows for context preservation.
2. **Asynchronous Webhooks:** Event-driven notifications via background threading upon successful ingestion.
3. **LLM Semantic Caching:** In-memory caching layer to reduce latency and API costs for repeated queries.
4. **Agentic Self-Correction:** Fallback mechanisms to optimize vector search queries when retrieval confidence is low.

## Local Setup

1. Start the pgVector database via Docker:
```bash
docker run --name pgvector-db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=enterprise_rag -p 5433:5432 -d pgvector/pgvector:pg16
