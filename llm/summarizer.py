def summarize(context_chunks, max_len=200):
    joined = " ".join(context_chunks)
    return joined[:max_len] + "..."
