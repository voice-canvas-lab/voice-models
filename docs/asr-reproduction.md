# Воспроизведение ASR-эксперимента ru-text-only-1.0.1

Команды ниже приведены для будущего запуска из корня репозитория после установки двух окружений по README. При подготовке публикации модели не запускались. Параметры моделей и нормализация не меняются. Полный аудиопрогон требует загрузки моделей и времени; для проверки опубликованных метрик достаточно отдельного блока README без моделей.

## Входы и проверка до моделей

Источник — точный снимок `examples/zhanna-v1/samples.jsonl` из PR №5, commit `a1dff1335109d12e5594391066d6a260ce54601b`. SHA256 и путь источника сохранены в `source.json`. Производные данные и правила лежат в `text-only-v1.0.1`; выбранные 151 sample_id и отдельные 12 вопросов — в `experiment-text-only-v1.0.1`.

Не запускайте старый `run_experiment.py` из локального плана: он привязан к прежним выходам и пилотным файлам. Не обновляйте старый freeze.json. Для нового запуска копируется конфигурация с новым выходным каталогом, а функция preflight проверяет её без загрузки моделей:

```powershell
New-Item -ItemType Directory -Path outputs -Force | Out-Null
.\.venv\Scripts\python.exe -B -c "import sys,json; from pathlib import Path; sys.path.insert(0,'scripts'); from run_pipeline import preflight; c=json.loads(Path('configs/pipeline-zhanna-text-only-v1.0.1.json').read_text(encoding='utf-8')); c['output']='outputs/reproduce-zhanna-text-only-v1.0.1'; p=Path('outputs/reproduce-zhanna-config.json'); p.open('x',encoding='utf-8').write(json.dumps(c,indent=2)); _,r,s,e=preflight(p); print('samples=',len(s),'expected_wavs=',len(s)*9)"
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; do not start models' }
```

Ожидание: 151 команда, 1359 WAV. Если конфигурация/выход уже существуют, выберите новые имена и согласованно замените их в следующих командах. Команды не перезаписывают прежние результаты.

Только для проверки воспроизводимости самого преобразования, без TTS/ASR:

```powershell
.\.venv\Scripts\python.exe -B scripts/spoken_text_text_only.py --source examples/zhanna-v1/samples.jsonl --output outputs/reproduce-derived-text-only-1.0.1
if ($LASTEXITCODE -ne 0) { throw 'Derivation failed' }
```

Результат: 163 строки, меняется только text; 151 готовая и 12 требующих решения. Входы не исправляются по args/context/action_id. Старый пересмотр 63 исключений — историческая информация, не критерий выбора нового набора.

## TTS, аудиоаудит и обязательный шлюз длительностей

```powershell
$Run = 'outputs/reproduce-zhanna-text-only-v1.0.1'
$Samples = 'examples/zhanna-v1/experiment-text-only-v1.0.1/ready_samples.jsonl'
$Models = 'docs/reports/zhanna-text-only-v1.0.1/artifacts/config.models.json'
if (Test-Path -LiteralPath $Run) { throw 'Choose a new run directory' }
New-Item -ItemType Directory -Path $Run | Out-Null
Copy-Item -LiteralPath $Samples -Destination "$Run/input_samples.jsonl"
Copy-Item -LiteralPath $Models -Destination "$Run/config.models.json"

.\.venv\Scripts\python.exe -B scripts/synthesize.py --samples "$Run/input_samples.jsonl" --config "$Run/config.models.json" --output "$Run/dataset"
if ($LASTEXITCODE -ne 0) { throw 'TTS failed; keep partial output' }
.\.venv\Scripts\python.exe -B scripts/check_dataset.py --dataset "$Run/dataset" --output "$Run/dataset/dataset_checks.json"
if ($LASTEXITCODE -ne 0) { throw 'Audio audit failed; do not start ASR' }

$Audit = Get-Content -LiteralPath "$Run/dataset/dataset_checks.json" -Raw -Encoding UTF8 | ConvertFrom-Json
$Audio = @(Get-Content -LiteralPath "$Run/dataset/audio_manifest.jsonl" -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json })
if (-not $Audit.passed -or $Audit.files -ne 1359 -or $Audio.Count -ne 1359) { throw 'Incomplete audio set' }
# The successful auditor has checked each manifest duration against actual WAV frames.
$TooLong = @($Audio | Where-Object { $_.duration_sec -gt 25 -or $_.duration_sec -le 0 })
if ($TooLong.Count -gt 0) {
    $TooLong | Select-Object audio_id,sample_id,voice,duration_sec,path | Format-Table -AutoSize
    throw 'STOP before BOTH ASR: invalid GigaAM short-form duration; no trimming or exclusions'
}
```

Silero V5: aidar/baya/xenia, seed 42, 48 kHz внутри TTS с полифазным переводом в mono PCM16 16 kHz. Параметры акцентов/ё и SHA256 зафиксированы. На каждый clean-файл добавляются варианты искусственного белого гауссовского шума SNR20/SNR10. Аудитор проверяет полный состав, формат, сигнал, клиппинг, связи ID, длительности, фактический SNR и побитовое воспроизведение шумовых вариантов. Это не проверка правильности произношения человеком.

## ASR и обе оценки

Выполняйте последовательно после успешного шлюза выше; загрузка и прогрев не входят в RTF.

```powershell
.\.venv\Scripts\python.exe -B scripts/transcribe.py --dataset "$Run/dataset" --config "$Run/config.models.json" --output "$Run/whisper"
if ($LASTEXITCODE -ne 0) { throw 'Whisper failed; preserve partial output and logs' }
.\.venv-gigaam\Scripts\python.exe -B scripts/transcribe_gigaam.py --dataset "$Run/dataset" --config "$Run/config.models.json" --output "$Run/gigaam"
if ($LASTEXITCODE -ne 0) { throw 'GigaAM failed; preserve partial output and logs' }

foreach ($Model in @('whisper', 'gigaam')) {
    .\.venv\Scripts\python.exe -B scripts/evaluate.py --dataset "$Run/dataset" --results "$Run/$Model" --output "$Run/evaluation-$Model"
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
    .\.venv\Scripts\python.exe -B scripts/evaluate_numeric.py --dataset "$Run/dataset" --results "$Run/$Model" --baseline "$Run/evaluation-$Model" --output "$Run/numeric-$Model"
    if ($LASTEXITCODE -ne 0) { throw 'Numerical evaluation failed' }
}
.\.venv\Scripts\python.exe -B scripts/compare_conditions.py --dataset "$Run/dataset" --whisper "$Run/evaluation-whisper" --gigaam "$Run/evaluation-gigaam" --output "$Run/comparison"
if ($LASTEXITCODE -ne 0) { throw 'Comparison failed' }
```

Оба адаптера сохраняют audio_sha256 в timings.jsonl. Сверьте их для всех audio_id и с фактическими WAV; совпадение числа записей само по себе не подтверждает идентичность входов:

```powershell
$Whisper = @{}
Get-Content -LiteralPath "$Run/whisper/timings.jsonl" -Encoding UTF8 | ForEach-Object { $r = $_ | ConvertFrom-Json; if ($Whisper.ContainsKey($r.audio_id)) { throw 'Duplicate Whisper ID' }; $Whisper[$r.audio_id] = $r.audio_sha256 }
$Giga = @{}
Get-Content -LiteralPath "$Run/gigaam/timings.jsonl" -Encoding UTF8 | ForEach-Object { $r = $_ | ConvertFrom-Json; if ($Giga.ContainsKey($r.audio_id)) { throw 'Duplicate GigaAM ID' }; $Giga[$r.audio_id] = $r.audio_sha256 }
if ($Whisper.Count -ne $Audio.Count -or $Giga.Count -ne $Audio.Count) { throw 'Incomplete timings' }
foreach ($a in $Audio) {
    $Hash = (Get-FileHash -LiteralPath (Join-Path "$Run/dataset" $a.path) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Whisper[$a.audio_id] -ne $Hash -or $Giga[$a.audio_id] -ne $Hash) { throw "Audio hash mismatch: $($a.audio_id)" }
}
```

Все гипотезы, времена, настройки и метрики остаются в новом `$Run`. Сводные обычные срезы — `comparison/`; отдельные числовые записи и тексты до/после преобразования — `numeric-*/`. Срезы числовой оценки завершённого эксперимента опубликованы в `docs/reports/zhanna-text-only-v1.0.1/artifacts/slices.csv`.

При HTTP 429 остановитесь и сохраните частичный результат; не меняйте ревизию на main. В опубликованном запуске GigaAM продолжили из локального snapshot с проверкой LFS SHA256 и Git blob ID, без повторения Whisper. Исторический локальный cache-only helper не является универсальным публичным загрузчиком; его доказательства сохранены в provenance.json. Повторный запуск после внешнего сбоя должен использовать новый каталог; не удаляйте предыдущие файлы.
