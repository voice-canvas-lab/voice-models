"""Точка запуска генерации текстового датасета."""

import argparse
import json
from pathlib import Path

from pipeline import PipelineConfig, run_pipeline, run_world_generation


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-answers", action="store_true", help="Совместимость со старыми запусками: проверки Python включены всегда")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/text.json", help="Путь к JSON-конфигурации")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--world-only", action="store_true", help="Сгенерировать intermediate/world.jsonl")
    modes.add_argument(
        "--use-prepared-exampls", "--use-prepared-examples", dest="use_prepared_examples",
        action="store_true", help="Загрузить аргументы из intermediate/draft_samples.jsonl",
    )
    args = parser.parse_args()
    config = PipelineConfig(
        **json.loads(args.config.read_text(encoding="utf-8"))
    )
    if args.world_only:
        run_world_generation(config, root=ROOT)
        return

    run_pipeline(
        config, root=ROOT,
        use_prepared_examples=args.use_prepared_examples,
    )


if __name__ == "__main__":
    main()
