"""Генерация разговорных команд с метками вместо значений аргументов."""

import json
import re
from string import Formatter

from jsonschema import Draft202012Validator, ValidationError

from command_form import validate_command_form


def generate_templates(action: dict, n_formulations: int, generate) -> list[dict]:
    parameters = action['parameters']
    required = set(parameters['required'])
    properties = parameters['properties']
    item_schema = {
        'type': 'object', 'properties': {'text': {'type': 'string', 'minLength': 1}},
        'required': ['text'], 'additionalProperties': False,
    }
    validator = Draft202012Validator(item_schema)
    prompt = (
        'Сформулируй разные короткие голосовые команды на русском для одного действия. '
        'Пиши разговорно, как человек обращается к приложению. Варьируй глаголы и порядок слов. '
        'Сам шаблон должен уже задавать естественную конструкцию для данного действия. '
        'Начни с русского глагола в повелительном наклонении. '
        'Формулируй прямую команду без вопросов, подтверждений и условий. '
        'Описание действия объясняет смысл, его не нужно копировать или пересказывать в команде. '
        'Вместо конкретных значений используй метки {имя_параметра}, включая тип объекта. '
        'Обязательные параметры включай в каждый шаблон; необязательные можешь включать или пропускать. '
        'Не добавляй другие метки, конкретные значения, действия или пояснения. '
        'Верни массив объектов с полем text.\n'
        f"Описание действия: {action['description']}\n"
        f'Схема параметров: {json.dumps(parameters, ensure_ascii=False)}'
    )
    accepted = {}
    error = ''
    for _ in range(3):
        remaining = n_formulations - len(accepted)
        schema = {'type': 'array', 'items': item_schema, 'minItems': remaining, 'maxItems': remaining}
        request = prompt + f'\nНужно шаблонов: {remaining}.'
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
        if len(accepted) == n_formulations:
            break
    if len(accepted) < n_formulations:
        print(f"{action['action_id']}: получено {len(accepted)}/{n_formulations} шаблонов.", flush=True)
    return list(accepted.values())
