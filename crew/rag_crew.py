from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from vectorstore.retriever import retrieve_chunks

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# --- Document Retriever Agent ---
retriever_agent = Agent(
    role="Document Retriever",
    goal="Retrieve relevant document chunks",
    backstory="Expert in semantic search",
    llm=llm
)

# --- Summarizer Agent ---
summarizer_agent = Agent(
    role="Summarizer",
    goal="Generate concise summaries",
    backstory="Expert in summarization",
    llm=llm
)

# --- Tasks ---
retrieve_task = Task(
    description="Retrieve relevant chunks for query",
    expected_output="Relevant document chunks",
    agent=retriever_agent,
    function=retrieve_chunks
)

def summarize_chunks_task(context):
    """
    context is expected to be a dict with retrieved chunks
    """
    chunks = context.get("retrieved_chunks", [])
    summary_length = context.get("summary_length", 200)

    summary_prompt = f"Summarize the following chunks in {summary_length} words:\n\n"
    for i, chunk in enumerate(chunks):
        summary_prompt += f"{i+1}. {chunk['text']}\n"

    # Wrap prompt in HumanMessage
    result = summarizer_agent.llm.generate([HumanMessage(content=summary_prompt)])[0].text
    return {"summary": result}

summarize_task = Task(
    description="Summarize retrieved chunks",
    expected_output="Final summary",
    agent=summarizer_agent,
    function=summarize_chunks_task
)

# --- Crew ---
rag_crew = Crew(
    agents=[retriever_agent, summarizer_agent],
    tasks=[retrieve_task, summarize_task],
    verbose=True
)
