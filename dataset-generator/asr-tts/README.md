# TTS → ASR → оценка

Модуль озвучивает команды, проверяет WAV, добавляет искусственный белый шум, запускает Whisper-small и GigaAM-v3 e2e_ctc и сравнивает гипотезы с эталоном. ASR получает только аудио, без исходного текста и разметки.

## Структура

| Путь от корня репозитория | Назначение |
|---|---|
| `dataset-generator/asr-tts/` | Рабочие скрипты и зависимости |
| `dataset-generator/asr-tts/tests/` | Тесты без загрузки моделей |
| `configs/asr-tts-pilot.json` | Минимальный запуск на четырёх командах |
| `configs/asr-tts.json` | Конфигурация опубликованного эксперимента |
| `data/v1/asr-tts/` | Исторический источник, произносимые тексты, соответствия и вопросы произношения |
| `data/v1/experiments/zhanna-text-only-v1.0.1/` | Сохранённые гипотезы, замеры, метаданные и метрики; без WAV |
| `outputs/` | Новые локальные результаты, исключены из Git |

`source_samples.jsonl` — точный снимок 163 команд старого коммита Жанны. Он не заменяет актуальный `data/v1/dataset/samples.jsonl` текстового генератора. `tts_ready_samples.jsonl` содержит 151 произносимый эталон, 12 вопросов сохранены в `pronunciation_review.jsonl`. Все ID выбранных команд доступны непосредственно в файле входов. Не подменяйте исторический источник новым датасетом при проверке старых метрик.

## Установка

Команды PowerShell выполняются из корня репозитория, Python 3.11, CPU:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r dataset-generator/asr-tts/requirements-tts.txt -r dataset-generator/asr-tts/requirements-asr.txt
py -3.11 -m venv .venv-gigaam
.\.venv-gigaam\Scripts\python.exe -m pip install -r dataset-generator/asr-tts/requirements-gigaam.txt
```

Окружения разделены из-за разных версий PyTorch. На Linux/macOS интерпретаторы находятся в `.venv/bin/python` и `.venv-gigaam/bin/python`; измените соответствующие пути в копии конфигурации. Зависимости опубликованного эксперимента проверялись на Windows CPU; установка на других платформах требует отдельной проверки. Первый запуск моделей скачивает веса.

## Тесты и маленький запуск

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s dataset-generator/asr-tts/tests -v
.\.venv\Scripts\python.exe dataset-generator/asr-tts/run_pipeline.py --config configs/asr-tts-pilot.json
```

Конвейер проверяет входы и окружения, затем последовательно выполняет синтез, аудит, обе ASR, обе обычные оценки и сравнение. При ошибке останавливается и сохраняет статус/логи. Выходной каталог должен быть новым. Конфигурация содержит относительные пути от корня репозитория. Полный исторический эксперимент с дополнительной числовой оценкой и общим шлюзом длительности воспроизводится [по этапам](../../docs/asr-reproduction.md), а не одной командой пилота.

## Оценка сохранённых гипотез без моделей

```powershell
$Artifacts = 'data/v1/experiments/zhanna-text-only-v1.0.1'
$Check = 'outputs/published-evaluation-check'
if (Test-Path -LiteralPath $Check) { throw 'Choose a new output directory' }
foreach ($Model in @('whisper', 'gigaam')) {
    .\.venv\Scripts\python.exe dataset-generator/asr-tts/evaluate.py --dataset "$Artifacts/dataset" --results "$Artifacts/$Model" --output "$Check/evaluation-$Model"
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
    .\.venv\Scripts\python.exe dataset-generator/asr-tts/evaluate_numeric.py --dataset "$Artifacts/dataset" --results "$Artifacts/$Model" --baseline "$Check/evaluation-$Model" --output "$Check/numeric-$Model"
    if ($LASTEXITCODE -ne 0) { throw 'Numeric evaluation failed' }
}
```

Для этой проверки нужны `jiwer==4.0.0` и его зависимости; веса, WAV и PyTorch не требуются. Метаданные без WAV позволяют воспроизвести метрики, но не проверить акустическое качество.

## Основные скрипты

- `spoken_text_text_only.py` и `number_words.py`: производный произносимый текст только из исходного `text`; названия, смысл и разметку не исправляют.
- `synthesize.py`, `audio_noise.py`, `check_dataset.py`: Silero, ресемплинг, шум с воспроизводимым seed и аудиоаудит.
- `transcribe.py`, `transcribe_gigaam.py`: распознавание и замеры после прогрева.
- `evaluate.py`: обычные корпусные WER/CER и общий RTF.
- `evaluate_numeric.py`, `numeric_equivalence.py`: отдельная ограниченная диагностика числовых форматов; не заменяет обычную оценку.
- `compare_conditions.py`: метрики по голосам, шуму, действиям и split.
- `run_pipeline.py`, `input_validation.py`, `recognition_config.py`: запуск и проверка входов/настроек.

Исторические версии произношения, фрагментный эксперимент и рабочие заметки исключены из текущего модуля. Старые артефакты сохраняют оригинальные пути и хеши как сведения о прошлом запуске; они не утверждают идентичность нынешнего кода. Извлечение числового помощника проверяется сравнением всех 163 преобразований. Новый эксперимент записывает новые хеши и результаты.
