"""Генерация разговорных команд с метками вместо значений аргументов."""

import json
import re
from math import ceil
from string import Formatter

from jsonschema import Draft202012Validator, ValidationError

from command_form import validate_command_form


def generate_templates(
    action: dict, n_formulations: int, generate, *,
    min_completion_ratio=0.8, max_generation_attempts=3,
) -> list[dict]:
    parameters = action['parameters']
    required = set(parameters['required'])
    properties = parameters['properties']
    item_schema = {
        'type': 'object', 'properties': {'text': {'type': 'string', 'minLength': 1}},
        'required': ['text'], 'additionalProperties': False,
    }
    validator = Draft202012Validator(item_schema)
    prompt = (
        'Составь короткие естественные команды на русском для указанного действия. '
        'Каждую команду начни с глагола в повелительном наклонении. '
        'Точно передавай смысл действия. Один и тот же подходящий глагол можно повторять. '
        'Вместо значений используй {имя_параметра} из схемы. '
        'Включай все обязательные параметры, необязательные — по необходимости. '
        'Верни до указанного числа корректных шаблонов без пояснений. '
        'Если естественных вариантов меньше, верни меньше. '
        'Формат ответа — массив объектов с полем text.\n'
        f"Действие: {action['description']}\n"
        f'Параметры: {json.dumps(parameters, ensure_ascii=False)}'
    )
    minimum = ceil(n_formulations * min_completion_ratio)
    accepted = {}
    error = ''
    for _ in range(max_generation_attempts):
        remaining = n_formulations - len(accepted)
        schema = {'type': 'array', 'items': item_schema, 'minItems': 1, 'maxItems': remaining}
        request = prompt + f'\nВерни до указанного числа шаблонов: {remaining}.'
        if accepted:
            request += '\nНе повторяй: ' + json.dumps(list(accepted), ensure_ascii=False)
        request_prompt = request + error
        try:
            response = json.loads(generate(request_prompt, schema))
            if not isinstance(response, list):
                raise ValueError('Ожидается массив шаблонов')
            for item in response:
                try:
                    validator.validate(item)
                    text = item['text'].strip()
                    validate_command_form(text)
                    parsed = list(Formatter().parse(text))
                    parts = [part for part in parsed if part[1] is not None]
                    names = {part[1] for part in parts}
                    if not required <= names <= set(properties) or any(part[2] or part[3] for part in parts):
                        raise ValueError('Используй обязательные параметры и только метки из схемы')
                    if any(re.search(r'[A-Za-z{}]', literal) for literal, _, _, _ in parsed):
                        raise ValueError('Пиши команду на русском, латиница допустима в именах меток')
                    key = ' '.join(text.split()).casefold()
                    if key:
                        accepted[key] = {'action_id': action['action_id'], 'parameters': sorted(names), 'text': text}
                    if len(accepted) == n_formulations:
                        break
                except (ValueError, ValidationError) as exc:
                    error = f'\nИсправь: {exc}'
        except (ValueError, ValidationError) as exc:
            error = f'\nИсправь ответ: {exc}'
        if len(accepted) >= minimum:
            break
    if len(accepted) < minimum:
        print(f"{action['action_id']}: получено {len(accepted)}/{n_formulations} шаблонов.", flush=True)
    return list(accepted.values())
