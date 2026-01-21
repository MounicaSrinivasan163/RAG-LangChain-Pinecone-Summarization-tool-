# crew/rag_crew.py

from langchain_openai import ChatOpenAI

# ---------------- LLM ----------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.0  # 🔒 reduce hallucination
)


def detect_intent(query: str):
    q = query.lower()
    if any(w in q for w in ["disadvantage", "drawback", "limitation", "negative"]):
        return "List ONLY the disadvantages or negative aspects explicitly mentioned."
    if any(w in q for w in ["advantage", "benefit", "merit"]):
        return "List ONLY the advantages or benefits explicitly mentioned."
    if any(w in q for w in ["steps", "process", "how to"]):
        return "Explain ONLY the steps explicitly described."
    if any(w in q for w in ["difference", "compare"]):
        return "Compare ONLY what is explicitly stated."
    return "Answer ONLY what is explicitly stated."


def summarize_chunks_task(context):
    """
    Strict RAG Answering:
    - Uses ONLY retrieved document content
    - Refuses if answer is not present in context
    """

    chunks = context.get("retrieved_chunks", [])
    summary_length = context.get("summary_length", 200)
    query = context.get("query", "")

    # 🚨 No retrieved chunks → hard refusal
    if not chunks:
        return {
            "summary": "No relevant information found in the provided documents."
        }

    # ---------------- Clean Context ----------------
    clean_context = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        else:
            text = str(chunk)

        if text and text.strip():
            clean_context.append(text.strip())

    if not clean_context:
        return {
            "summary": "No relevant information found in the provided documents."
        }

    context_text = "\n\n".join(clean_context)

    intent_instruction = detect_intent(query)

    # ----------------  PROMPT ----------------
prompt = f"""
You are a document-grounded AI assistant.

CRITICAL RULES (must follow):
- Answer using ONLY the information present in the Context.
- Do NOT use external knowledge or assumptions.
- Do NOT mention the context, documents, or what is missing.
- Do NOT add disclaimers such as:
  "no other information is available",
  "explicitly mentioned",
  "not provided in the context",
  or similar phrases.
- If the answer is not present at all, respond ONLY with:
  "No relevant information found in the provided documents."

User Question:
{query}

Context:
{context_text}

Answering Instructions:
{intent_instruction}

Answer Style Rules:
- Provide ONLY the requested information
- Do NOT explain omissions
- Do NOT add concluding or meta statements
- Be concise and factual
- Use bullet points or short paragraphs if helpful
- Max length: {summary_length} words

Final Answer:
"""
 

    try:
        response = llm.invoke(prompt)
        summary = response.content.strip()

        # 🔒 Final safety net (post-check)
        if not summary or "no relevant information" in summary.lower():
            return {
                "summary": "No relevant information found in the provided documents."
            }

    except Exception as e:
        summary = f"Error generating response: {e}"

    return {"summary": summary}
