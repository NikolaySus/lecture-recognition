# Проверка оценщиков полезности каналов

2026-10-07. Пользователь выбрал историческую раскладку и склейку current после проверки S001–S012. Это отдельная серия; regular-m20-c2/contextual не перезаписывается и не используется как основа селекторов.

Текущая серия: `.lecture-cache/channel-research/83d3f707de5afcc1`.
Базовые контроли воспроизведены на R001–R008: CTC combination 18/246 ошибок, RNNT left 19/246. У каждой модели 22 исторических чанка. Исходные ASR-тексты чанков и aligned-блоки сохраняются в исходном кэше и в результатах диагностики; итоговая склейка не заменяет их.

Проверка использует только разработочные R001–R008. Перекрывающиеся карточки оцениваются через три общие группы, а не суммированием карточных ошибок. S001–S012 уже просмотрены при проверке раскладок и на этом этапе не используются.

## Что измеряем

- GigaAM confidence: Gibbs и Tsallis, одинаковые определения для обоих каналов, исключение blank; агрегирование по словам. Трасса CTC получена по greedy-выходу posterior, а не является вероятностью bias/beam-гипотезы combination. Это прокси полезности канала; его качество необходимо проверять по итоговому WER.
- CTC-margin: вероятности обеих конкурирующих ASR-гипотез на обоих каналах через CTC forward. Для RNNT-гипотез scorer тоже CTC. Непредставимые tokenizer-ом тексты дают unavailable, не подменяются нулём.
- Brouhaha: speech probability, SNR и C50 отдельно для left/right, агрегированные на речевых кадрах. Высокие SNR/C50 не гарантируют, что слышен именно лектор; проверяем связь с качеством его распознавания.

После diagnose команда develop проверяет заранее заданные пороги, сравнивает селекторы с фиксированными каналами, считает WER/CER, числа/отрицания, решения каналов и сравнение с групповым oracle. Это разработочная проверка, а не независимое доказательство улучшения. Групповой oracle грубый: смешанный выбор чанков может оказаться лучше любого одного канала на всей группе.

## Запуск на хосте

Окружения GigaAM и Brouhaha уже существуют. Из корня проекта сначала загрузить официальный pinned checkpoint:

```bash
.venv/bin/python scripts/setup_brouhaha.py
```

Затем собрать диагностики:

```bash
.venv/bin/python -u scripts/benchmark_channel_research.py diagnose --retry-failed
```

После успешного сбора проверить оценщики на разработочном наборе:

```bash
.venv/bin/python -u scripts/benchmark_channel_research.py develop --retry-failed
```

CLI наследует fixed_layout=historical и merger=current из latest.json. Для явного указания выбранной серии к diagnose/develop можно добавить `--fixed-layout historical --merger current`. Для восстановления подготовительного шага: `historical-baseline --retry-failed`. Команда chunk-study в серии с фиксированной раскладкой запрещена, чтобы не заменить выбор новым поиском.

Setup скачивает официальный GitHub revision 9132cbe62ac78f90abdbc21bcf6ec6cfe9bb4891, проверяет Git blob SHA1 (включая уже скачанные файлы), записывает SHA256 checkpoint в experiments/brouhaha/model.json; worker повторно проверяет SHA256 перед загрузкой. Источник: https://github.com/marianne-m/brouhaha-vad .

В текущей среде сетевые запросы запрещены (Operation not permitted), CUDA недоступна. Подготовка исторических контролей завершена; diagnose корректно остановлен как blocked, без записи ложного отказа модели. Checkpoint не скачан, численных результатов оценщиков пока нет. Если Brouhaha отсутствует, ASR-диагностики сохраняются, develop выдаёт только предварительные ASR-кандидаты и не допускает полного freeze.

Артефакты: chunk-frozen.json, diagnostics/*/traces/*.npz, diagnose.json, brouhaha/{left,right}.json, develop.json, status.json. Просмотр прогресса:

```bash
.venv/bin/python scripts/benchmark_channel_research.py progress
```

28 профильных тестов прошли; Ruff чист. Независимый тест и production-допуск на этом этапе не запускаются.
