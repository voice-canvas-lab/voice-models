"""Генерация JSON по схеме через Ollama."""

from ollama import chat


def generate_json(
    prompt: str,
    schema: dict,
    *,
    model: str,
    seed: int,
    temperature: float,
    num_predict: int,
    num_ctx: int,
) -> str:
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"seed": seed, "temperature": temperature, "num_predict": num_predict, "num_ctx": num_ctx},
        stream=False,
    )
    if response.done_reason == "length":
        raise ValueError("Ollama достигла лимита длины ответа; увеличьте num_predict")
    if not response.message.content:
        raise ValueError("Ollama вернула пустой ответ")
    return response.message.content
