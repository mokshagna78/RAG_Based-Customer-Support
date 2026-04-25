from langgraph.graph import StateGraph

from modules.generator import generate_answer
from modules.router import route
from modules.hitl import escalate_to_human


def build_graph(llm, retriever):
    def process(state):
        query = state["query"]

        #Retrieve documents
        docs = retriever.invoke(query)

        #Filter relevant chunks (important)
        query_keywords = query.lower().split()
        filtered_docs = []

        for doc in docs:
            content = doc.page_content.lower()
            if any(word in content for word in query_keywords):
                filtered_docs.append(doc)

        #Fallback if filtering removes too much
        if len(filtered_docs) < 2:
            filtered_docs = docs[:3]

        #Generate answer using filtered context
        answer = generate_answer(llm, query, filtered_docs)

        return {
            "query": query,
            "docs": filtered_docs,
            "answer": answer
        }
    def decide(state):
        return route(state["query"], state["docs"], state["answer"])
    def answer_node(state):
        return {"output": state["answer"]}
    def escalate_node(state):
        msg = escalate_to_human(state["query"])
        return {"output": msg}
    graph = StateGraph(dict)
    graph.add_node("process", process)
    graph.add_node("answer", answer_node)
    graph.add_node("escalate", escalate_node)
    graph.set_entry_point("process")
    graph.add_conditional_edges(
        "process",
        decide,
        {
            "answer": "answer",
            "escalate": "escalate"
        }
    )
    return graph.compile()