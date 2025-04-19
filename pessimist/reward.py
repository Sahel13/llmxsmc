tragic_keywords = [
    "died", "kill", "maim", "alone", "forgotten", "lost", "tears", "funeral", "grave", "sorrow", "regret", "sad"
]

def tragicness_score(text: str) -> float:
    count = sum(word in text.lower() for word in tragic_keywords)
    return count / len(tragic_keywords)
