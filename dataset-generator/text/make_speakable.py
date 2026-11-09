"""Подготовка команд к произнесению с проверкой каждого пакета."""

import json

from jsonschema import Draft202012Validator, ValidationError

from command_form import validate_command_form


def make_speakable(examples, actions, generate, *, log_every=10):
    batch_size = 4
    actions_by_id = {action['action_id']: action for action in actions}
    item_schema = {
        'type': 'object',
        'properties': {'id': {'type': 'integer'}, 'text': {'type': 'string', 'minLength': 1}},
        'required': ['id', 'text'], 'additionalProperties': False,
    }
    validator = Draft202012Validator(item_schema)
    samples = []
    next_log = log_every
    for start in range(0, len(examples), batch_size):
        pending = dict(enumerate(examples[start:start + batch_size], start))
        error = ''
        for _ in range(3):
            commands = [
                {
                    'id': index, **example,
                    'action': actions_by_id[example['action_id']]['description'],
                    'parameters': actions_by_id[example['action_id']]['parameters'],
                }
                for index, example in pending.items()
            ]
            prompt = (
                'Сделай текст естественной голосовой командой на русском. '
                'Формулируй прямую команду без вопросов, подтверждений и условий. '
                'Исправь падежи, согласование и предлоги, сохрани действие и смысл каждого аргумента. '
                'В именах файлов убери расширения, включая составные; в названиях замени подчёркивания пробелами. '
                'Эти изменения и кавычки вокруг названий допустимы; остальные части значений сохраняй. '
                'Названия с вопросительным знаком заключай в кавычки. '
                'Меняй только text. Не произноси сведения из контекста, которых не было в черновике. '
                'Если исправления не нужны, оставь текст как есть. '
                'Верни массив объектов id и text для всех команд, сохранив id.\n'
                f'Команды: {json.dumps(commands, ensure_ascii=False)}'
            )
            request_prompt = prompt + error
            schema = {'type': 'array', 'items': item_schema, 'minItems': len(pending), 'maxItems': len(pending)}
            try:
                result = json.loads(generate(request_prompt, schema))
                if not isinstance(result, list):
                    raise ValueError('Ожидается массив команд')
                for item in result:
                    try:
                        validator.validate(item)
                        text = item['text'].strip()
                        if item['id'] not in pending or not text:
                            continue
                        validate_command_form(text)
                        example = pending.pop(item['id'])
                        samples.append({**example, 'text': text})
                    except (ValueError, ValidationError) as exc:
                        error = f'\nИсправь формат ответа: {exc}'
            except (ValueError, ValidationError) as exc:
                error = f'\nИсправь формат ответа: {exc}'
            if not pending:
                break
        processed = min(start + batch_size, len(examples))
        if processed >= next_log or processed == len(examples):
            print(f'Подготовка речи: обработано {processed}/{len(examples)}, получено {len(samples)} команд.', flush=True)
            next_log = (processed // log_every + 1) * log_every
    return samples

