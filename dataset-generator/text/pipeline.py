"""Генерация аргументов, команд и сохранение текстового датасета."""

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile

from llm import generate_json
from generate_world import generate_world, load_object_types, load_world
from make_speakable import make_speakable
from generate_templates import generate_templates
from prepare_examples import prepare_examples
from split import assign_splits
from jsonschema import Draft202012Validator, ValidationError


@dataclass(frozen=True)
class PipelineConfig:
    inputs_dir: str
    run_dir: str
    n_texts_per_template: int
    n_templates_per_action: int
    max_parameter_length: int
    split_ratios: list[float]
    seed: int
    model: str
    temperature: float
    num_predict: int
    num_ctx: int
    log_every: int
    n_objects_per_type: int = 100

    def __post_init__(self) -> None:
        for name in (
            "n_texts_per_template", "n_templates_per_action", "max_parameter_length",
            "num_predict", "num_ctx", "log_every", "n_objects_per_type",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} должен быть положительным целым числом")
        if type(self.seed) is not int:
            raise ValueError("seed должен быть целым числом")
        if type(self.temperature) not in (int, float) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature должна быть числом от 0 до 2")
        if (
            not isinstance(self.split_ratios, list)
            or len(self.split_ratios) != 3
            or any(type(r) not in (int, float) or not 0 <= r <= 1 for r in self.split_ratios)
            or abs(sum(self.split_ratios) - 1) > 1e-9
        ):
            raise ValueError("split_ratios: три доли train/validation/test с суммой 1")
        for name in ("inputs_dir", "run_dir", "model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} должен быть непустой строкой")


def load_actions(path: Path) -> list[dict]:
    actions = []
    ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        action = json.loads(line)
        if not isinstance(action, dict):
            raise ValueError(f"Строка {line_number}: ожидается объект действия")
        
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id.strip() or action_id in ids:
            raise ValueError(f"Строка {line_number}: пустой или повторяющийся action_id")
        
        if not isinstance(action.get("description"), str) or not action["description"].strip():
            raise ValueError(f"{action_id}: отсутствует описание")
        
        schema = action.get("parameters")
        if (
            not isinstance(schema, dict)
            or schema.get("type") != "object"
            or not isinstance(schema.get("properties"), dict)
            or not isinstance(schema.get("required"), list)
            or schema.get("additionalProperties") is not False
        ):
            raise ValueError(f"{action_id}: неверная структура parameters")
        
        if any(not isinstance(key, str) or key not in schema["properties"] for key in schema["required"]):
            raise ValueError(f"{action_id}: required ссылается на неизвестный параметр")
        
        for name, prop in schema["properties"].items():
            if (
                not isinstance(prop, dict) or "type" not in prop
                or not isinstance(prop.get("description"), str) or not prop["description"].strip()
            ):
                raise ValueError(f"{action_id}.{name}: необходимы type и description")
        Draft202012Validator.check_schema(schema)
            
        ids.add(action_id)
        actions.append(action)

    if not actions:
        raise ValueError("Каталог действий пуст")
    return actions


def configured_generator(config: PipelineConfig):
    return partial(
        generate_json, model=config.model, seed=config.seed,
        temperature=config.temperature, num_predict=config.num_predict,
        num_ctx=config.num_ctx,
    )


def load_catalogs(config: PipelineConfig, *, root: Path) -> tuple[list[dict], dict[str, str]]:
    inputs_dir = root / config.inputs_dir
    actions_path = inputs_dir / "actions.jsonl"
    actions = load_actions(actions_path)
    object_types = load_object_types(inputs_dir / "object_types.jsonl")
    for action in actions:
        properties = action["parameters"]["properties"]
        for field, field_schema in properties.items():
            if "x-world-object-type" in field_schema:
                value = field_schema["x-world-object-type"]
                if not isinstance(value, str) or value not in object_types:
                    raise ValueError(
                        f"{actions_path}: {action['action_id']}.{field}: "
                        f"тип {value!r} отсутствует в object_types.jsonl"
                    )
        parameter = properties.get("object_type")
        if parameter is None:
            continue
        label = f"{actions_path}: {action['action_id']}.object_type"
        if parameter["type"] != "string":
            raise ValueError(f"{label}: ожидается параметр типа string")
        allowed_types = list(parameter.get("enum", []))
        if "const" in parameter:
            allowed_types.append(parameter["const"])
        if not allowed_types:
            raise ValueError(f"{label}: задайте допустимые типы в enum или const")
        for value in allowed_types:
            if not isinstance(value, str) or value not in object_types:
                raise ValueError(f"{label}: тип {value!r} отсутствует в object_types.jsonl")
    return actions, object_types


def run_world_generation(config: PipelineConfig, *, root: Path) -> None:
    _, object_types = load_catalogs(config, root=root)
    save_generated_world(
        config, object_types, configured_generator(config),
        root / config.run_dir / "intermediate" / "world.jsonl",
    )


def save_generated_world(config: PipelineConfig, object_types, generate, path: Path) -> list[dict]:
    print(f"Генерация мира: модель {config.model}.", flush=True)
    objects = generate_world(
        object_types,
        generate,
        n_objects_per_type=config.n_objects_per_type,
        max_name_length=config.max_parameter_length,
    )
    store_in_file(path, objects)
    return objects


def run_pipeline(
    config: PipelineConfig, *, root: Path,
    use_prepared_examples: bool = False,
) -> list[dict]:
    print("Загрузка и проверка каталогов действий и типов объектов...", flush=True)
    actions, object_types = load_catalogs(config, root=root)
    run_dir = root / config.run_dir
    intermediate_dir = run_dir / "intermediate"
    drafts_path = intermediate_dir / "draft_samples.jsonl"
    generate = configured_generator(config)
    world = []
    if not use_prepared_examples:
        world_path = intermediate_dir / "world.jsonl"
        try:
            world = load_world(world_path, object_types)
            print(f"Загружен мир: {len(world)} объектов из {world_path}.", flush=True)
        except FileNotFoundError:
            world = save_generated_world(config, object_types, generate, world_path)

    templates = []
    print(f"Генерация шаблонов: {len(actions)} действий, модель {config.model}.", flush=True)
    for index, action in enumerate(actions, 1):
        print(f"Шаблоны: действие {index}/{len(actions)} — {action['action_id']}...", flush=True)
        templates.extend(generate_templates(action, config.n_templates_per_action, generate))
    if not templates:
        raise ValueError("Модель не сгенерировала допустимых шаблонов.")
    store_in_file(intermediate_dir / "command_templates.jsonl", templates)
    examples = []
    rng = Random(config.seed)
    if use_prepared_examples:
        print(f"Загрузка готовых аргументов: {drafts_path}", flush=True)
        actions_by_id = {action["action_id"]: action for action in actions}
        for line_number, line in enumerate(drafts_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                example = json.loads(line)
                if not isinstance(example, dict) or not isinstance(example.get("action_id"), str):
                    raise ValueError("Ожидается объект с action_id")
                if example["action_id"] not in actions_by_id:
                    raise ValueError(f"Неизвестное действие: {example['action_id']}")
                Draft202012Validator(actions_by_id[example["action_id"]]["parameters"]).validate(example.get("args"))
                context = example.get("context")
                if (
                    not isinstance(context, dict)
                    or not isinstance(context.get("objects"), list)
                    or not isinstance(context.get("selected_object_ids"), list)
                ):
                    raise ValueError("context должен содержать списки objects и selected_object_ids")
            except (ValueError, ValidationError) as exc:
                raise ValueError(f"{drafts_path}, строка {line_number}: {exc}") from exc
            examples.append({key: example[key] for key in ("action_id", "args", "context")})
        if not examples:
            raise ValueError(f"Файл заготовок пуст: {drafts_path}")
    prepared = examples if use_prepared_examples else None
    examples = []
    print("Подбор значений и подстановка в шаблоны...", flush=True)
    for index, action in enumerate(actions, 1):
        print(f"Примеры: действие {index}/{len(actions)} — {action['action_id']}...", flush=True)
        action_templates = [template for template in templates if template["action_id"] == action["action_id"]]
        examples.extend(prepare_examples(
            action, action_templates, config.n_texts_per_template, world=world, rng=rng,
            prepared_examples=prepared, max_parameter_length=config.max_parameter_length,
        ))
    if not examples:
        raise ValueError("Не удалось заполнить шаблоны аргументами.")
    if not use_prepared_examples:
        store_in_file(drafts_path, examples)
    print(f"Подготовка текста к произнесению: {len(examples)} команд...", flush=True)
    samples = make_speakable(
        examples, actions, generate, log_every=config.log_every,
    )
    if not samples:
        raise ValueError("Подготовка речи не вернула ни одной допустимой команды.")
    print(f"Удаление дублей и распределение {len(samples)} формулировок по выборкам...", flush=True)
    samples = assign_splits(samples, config.split_ratios, config.seed)
    store_in_file(run_dir / "dataset" / "samples.jsonl", samples)
    print("Генерация завершена.", flush=True)
    return samples


def store_in_file(output_path: Path, examples: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file = NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent,
        prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
    )
    temporary = Path(file.name)
    try:
        with file:
            for example in examples:
                file.write(json.dumps(example, ensure_ascii=False) + "\n")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Сохранено строк: {len(examples)}. Файл: {output_path}", flush=True)
