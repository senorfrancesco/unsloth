# Plan

Делаем управляемый форк `unsloth`, который сохраняет совместимость с `upstream`, добавляет отдельную административную поверхность для `RAG Upload`, `Collections` и `Jobs`, а затем упаковывается в собственный контейнер для серверной обработки. Базовый принцип: минимизировать прямые правки в существующих потоках `Studio` и `Export`, а новые возможности добавлять через отдельные маршруты, модули и API.

## Scope
- In: стратегия сопровождения форка и синхронизации с `upstream`, архитектура новых административных поверхностей, точки правок во `frontend` и `backend`, визуальные правила для согласованных расширений интерфейса, контейнеризация и серверный запуск форка.
- Out: полный редизайн текущего `Studio`, редизайн `training`-поверхности, замена существующих потоков `training` и `export`, реализация `distributed training`, финальный production rollout без промежуточной проверки.

## Action items
[x] Разобрать архитектурные материалы `production_multiagent_architecture(2).html`, `unsloth_fork_tz.md` и `unsloth_fork_rag_panel.html` и свести их в единое понимание границ форка.
[x] Снять карту текущих точек правок в `studio/frontend` и `studio/backend`, включая `studio-page.tsx`, `export-page.tsx`, `data-recipes`, `app-sidebar.tsx`, `routes/*` и существующие `FastAPI`-маршруты.
[x] Зафиксировать текущий визуальный язык интерфейса по `studio/frontend/src/index.css`, `components/section-card.tsx`, `components/app-sidebar.tsx` и ключевым страницам `studio`, `export`, `data-recipes`.
[x] Исследовать текущую `RAG`-архитектуру соседнего `open-webui`: `knowledge`-модель, `retrieval`-роуты, векторную фабрику, схему `reindex`, режимы `Focused Retrieval` / `Full Context` и UI-потоки привязки `knowledge base` к чату и модели.
[x] Собрать внешние лучшие практики по `Qdrant`, `pgvector`, `Milvus`, `Weaviate`, `Chroma`, `Docling`, `Tika` и настройкам `Open WebUI`, чтобы зафиксировать приоритеты поддержки, модульность провайдеров и требования к схеме коллекций.
[x] Исследовать `agent-navigator-pro` как второй локальный референс по `Qdrant RAG`: `KnowledgeBaseStoreProtocol`, `QdrantKnowledgeBaseStore`, ingestion helper, merged retrieval `knowledge + session overlay`, persisted chunk embeddings и операторское разделение пространств имён между backend `session RAG` и native `Open WebUI Knowledge`.
[x] Проверить текущий графовый интерфейс `unsloth` для `data-recipes` и `recipe-studio`, чтобы понять, нужен ли похожий `board`/`canvas` для `RAG`-настроек, и зафиксировать точки повторного использования `@xyflow/react`, узлов, боковых панелей и плавающих контролов.
[x] Добавить `upstream`-репозиторий и завести чистую зеркальную ветку `upstream-main`, которая всегда совпадает с `upstream/main`.
[x] Закрепить долгоживущую ветку интеграции форка как текущую `develop` и вести новые изменения поверх неё, а не поверх зеркала `upstream`.
[x] Утвердить информационную архитектуру: оставить `/studio` для обучения без редизайна, `/export` для экспорта, `/data-recipes` для генерации датасетов и добавить отдельный административный маршрут `/admin/rag` для `RAG Upload`, `Collections` и `Jobs`.
[x] Зафиксировать матрицу поддержки `RAG`-бэкендов для первого этапа: рабочий `Qdrant`, design-only `pgvector`; second-wave `Milvus` и `Weaviate`; compatibility/dev-only `Chroma`; не брать в первый этап облачно-специфичные `Pinecone`, `S3Vector`, `Oracle 23ai`, `OpenSearch`, `Elasticsearch`.
[x] Создать новый модуль интерфейса `studio/frontend/src/features/admin-rag` и новые маршруты в `studio/frontend/src/app/routes/*` вместо перегрузки `studio/frontend/src/features/studio/studio-page.tsx`.
[x] Расширить навигацию в `studio/frontend/src/components/app-sidebar.tsx`, не ломая текущую схему `Studio` / `Export` / `Data Recipes` и не меняя сам сценарий `training`.
[x] Ввести отдельную продуктовую сущность `RAG dataset` или `RAG corpus`, отличную от `training dataset`: это набор документов или нормализованных текстов для индексации, версионирования, синхронизации с `Open WebUI` и повторного `reindex`.
[x] Спроектировать и добавить отдельные backend-маршруты для ingestion, коллекций и заданий рядом с `studio/backend/routes/*`: `overview`, `providers`, `collections`, `jobs`, `ingestion`, `reindex`, `sync-open-webui`.
[x] Вынести доменную логику новых задач в отдельные сервисы и модели рядом с `studio/backend/core/*`, `studio/backend/models/*`, `studio/backend/state/*`, чтобы избежать глубоких правок в `training.py`, `export.py`, `inference.py` и базовых сценариях запуска.
[x] Ввести реестр провайдеров для `extractor`, `ocr`, `chunker`, `embedder`, `reranker` и `vector_store`, чтобы новые движки добавлялись через декларативную конфигурацию, а не через ветвления по всему `backend`.
[x] Реализовать модульный каталог `RAG`: `modules/catalog`, проверки доступности, allowlist-установку Python-пакетов, локальную установку из `.whl` или `wheelhouse`, хранение статусов модулей в `SQLite`.
[x] Ввести экземпляры `RAG`-модулей: конкретный локальный путь модели, `model_id`, системный бинарник или service URL, которые можно тестировать и привязывать к профилю ingestion.
[x] Ввести сущность профиля коллекции или профиля ingestion: набор настроек `extractor`/`ocr`/`chunking`/`embedding`/`reranking`/`storage`, который можно применять к новым коллекциям и версиям индекса.
[x] Расширить профиль ingestion флагами включения `extractor` / `ocr` / `reranker` и ссылками на конкретные экземпляры `extractor`, `ocr`, `embedder`, `reranker`, сохранив старые поля как fallback.
[x] Ввести backend protocol boundary наподобие `KnowledgeBaseStoreProtocol`, чтобы `execution` и `retrieval` зависели от интерфейса хранилища, а не от конкретного `Qdrant`-клиента или схемы `SQLite`.
[x] Развести логическую коллекцию и физический индекс: хранить отдельную запись `knowledge_base` / `collection`, а физические индексы и пространства имён делать сменяемыми по версии эмбеддера и рецепта чанкинга.
[x] Зафиксировать схему метаданных чанка для совместимости с `Open WebUI`: `knowledge_base_id`, `file_id`, `hash`, `name`, `source`, `created_by`, `content_type`, `embedding_config`, `chunk_recipe`, `extractor`, `ocr_engine`, `chunk_index`, `start_index`, `page`, `section_path`, `language`.
[x] Спроектировать стратегию `reindex`: пересчёт векторов без повторного OCR при неизменном извлечённом тексте, и полный `rebuild` только при смене `extractor` / `ocr` / нормализованного содержимого.
[ ] Спроектировать стратегию zero-downtime-переключения индексов для `Qdrant` и `Weaviate` через `aliases`, а для `pgvector` через версионирование `collection_name` и атомарную смену активной проекции.
[ ] Спроектировать явное разделение `session` и `knowledge` областей в одном поисковом слое: отдельные `source_scope`, `thread_id`, `workspace_id`, `expires_at`, политика `session overlay`, дедуп по нормализованному тексту и deterministic tie-break между `session` и `knowledge`.
[ ] Добавить слой совместимости с `Open WebUI`: экспорт или синхронизацию `knowledge`-записей, `knowledge_file`-связей, служебной коллекции `knowledge-bases`, а также нормализацию старых полей `collection_name` / `collection_names` к новому `model.meta.knowledge`.
[ ] Добавить операторские и диагностические маршруты по `Qdrant`: сводка по активным пространствам имён, счётчикам `session` / `knowledge`, префиксам `Open WebUI`, проблемам смешения контуров и признакам нарушения namespace separation.
[x] Реализовать интерфейсные панели `RAG Upload`, `Collections` и `Jobs` в текущем стиле: `SectionCard`, нейтральная палитра, компактная типографика, мягкие `ring`-границы, статусные точки и без тяжёлого редизайна.
[x] Добавить вкладку `Modules` в `/admin/rag`: карточки каталога модулей, статусы установки, кнопки `Check` / `Install`, форму локальной установки и форму создания экземпляров модулей.
[x] Привести `Modules` к логике выбора моделей: поиск, секции `Installed / Configured`, `Available`, `Missing`, компактные строки в стиле текущего выбора LLM-моделей и переиспользуемый `RAGModuleSelector`.
[x] Показывать в форме создания `RAG collection` read-only preview выбранного ingestion-профиля: фактические `extractor`, `OCR`, `embedder`, `reranker`, путь или пакет, статус и предупреждения по отсутствующим модулям.
[ ] Спроектировать двухрежимный UI для `RAG`: быстрый сценарий через карточки и формы для типового потока, а также расширенный `board`/`canvas` для сложных ingestion-схем, где пользователь сам собирает стадии обработки и проверяет связи между ними.
[ ] Повторно использовать паттерны текущего `recipe-studio`, а не изобретать новый визуальный язык: `@xyflow/react`, узлы с тональностями и иконками, боковую `sheet`-панель настроек, плавающие контролы запуска и проверки, режимы просмотра выполнения.
[ ] Зафиксировать набор обязательных карточек и соответствующих узлов для `RAG`-контура: `RAG Dataset`, `Backend Target`, `Collection`, `Extractor`, `OCR`, `Chunking`, `Embedding`, `Reranker`, `Sync to Open WebUI`, `Jobs / Runs`, `Validation`, `Preview / Sample Retrieval`.
[x] Спроектировать, какие сценарии должны работать без доски: создать коллекцию, выбрать `RAG`-бэкенд, загрузить корпус, запустить индексацию, дозагрузить документы, пересобрать индекс, синхронизировать с `Open WebUI`.
[ ] Спроектировать, какие сценарии требуют доски: комбинированные цепочки `extract -> normalize -> OCR -> chunk -> embed -> index -> sync`, альтернативные ветки провайдеров, предпросмотр текста после извлечения, ручная валидация и повторный запуск этапов.
[x] Отдельно описать карточки списка коллекций: имя, `backend`, активная проекция индекса, профиль ingestion, число документов, число чанков, эмбеддер, статус синхронизации с `Open WebUI`, последний `reindex`, ошибки последнего задания.
[x] Добавить `RAG Collection Inspector`: обзор активной проекции, чтение sample chunks из `Qdrant` без сырых векторов, тестовый retrieval по query embedding и распределения по документам, размерам чанков и статусам индексации.
[x] Обработать upload-файлы для `RAG dataset`: пакетный выбор файлов, выбор папки и архивы `zip`/`tar`/`gz`/`rar` в UI и backend, явный статус обработки dataset, видимый `last_error` при ошибке извлечения и метаданные обработки у документов.
[x] Зафиксировать решение по выбору папки для `RAG upload`: это клиентская загрузка через браузерный `webkitdirectory`, где UI передаёт файлы с относительными путями, а не серверный `browse-folders` как в нативном выборе моделей. Серверный импорт папки с машины backend оставить отдельной будущей задачей, если он понадобится.
[x] Доработать управление `RAG Modules`: выбор модуля и пакета через поисковое меню в стиле выбора моделей, выбор локальной папки модели через серверный `browse-folders`, скрытие `Install` для доступных модулей и корзина для удаления записи модели или Python-пакета.
[x] Отдельно описать карточки списка заданий: тип задания, коллекция, инициатор, длительность, текущий этап, прогресс по стадиям, журнал, предупреждения, возможность открыть детали и перейти к связанному `RAG dataset`.
[ ] Добавить конфигурационные флаги и переменные окружения для новых панелей, фоновых задач и интеграций, чтобы можно было быстро отключать кастомные возможности без отката кода.
[ ] Вынести контейнеризацию в отдельный слой `deploy/` или `ops/`, не смешивая `Dockerfile`, `compose` и серверные скрипты с базовой логикой продукта.
[ ] Подготовить собственный `Dockerfile` для форка с запуском через штатный `unsloth studio`, а постоянные каталоги `~/.unsloth/studio`, `~/.cache/huggingface`, выходные артефакты и базы данных вынести в тома.
[ ] Собрать сценарий серверного запуска и локального воспроизведения: `./install.sh --local`, `unsloth studio -H 0.0.0.0 -p 8888`, контейнерный запуск и шаги обновления форка после синхронизации с `upstream`.
[ ] Добавить проверки: smoke-тест маршрутов, базовую проверку новых API, проверку сборки `studio/frontend`, проверку запуска `backend` и отдельный сценарий обновления форка после `upstream sync`.
[ ] Добавить интеграционные проверки совместимости с `Open WebUI`: создание `knowledge base`, добавление файла, `reindex`, поиск по `knowledge-bases`, отображение в списках вложений модели и чата.
[ ] Подготовить короткий документ сопровождения форка: как синхронизировать `upstream`, как обновлять контейнер, какие тома обязательны и какие файлы считаются основными точками конфликта.

## Research findings
### Приоритеты поддержки `RAG`
- `Qdrant` — основной целевой бэкенд для форка. Он уже присутствует в архитектурных материалах, хорошо стыкуется с self-hosted сценарием, поддерживает `payload`-фильтры, отдельные `payload indexes`, `named vectors`, гибридные запросы dense+sparse и атомарные `aliases` для безостановочного перевыпуска индекса.
- `pgvector` — второй основной бэкенд для простых серверных установок, где уже есть `Postgres`. Плюсы: `ACID`, резервное копирование и восстановление средствами `Postgres`, `JOIN`, единый стек хранения. Минусы: нет нативной сущности коллекции, поэтому логическую коллекцию надо моделировать через `collection_name`, индексы и версионирование.
- `Milvus` — отложить на второй этап как бэкенд для крупных установок. Он силён в многоарендности, поддерживает стратегии `database` / `collection` / `partition` / `partition key`, фильтры и multi-vector hybrid search, но заметно тяжелее в сопровождении.
- `Weaviate` — также второй этап. Сильные стороны: встроенные `collections`, многоарендность, `hybrid search`, `collection aliases`. Это хороший вариант для инсталляций, где нужна более богатая схема самого векторного слоя.
- `Chroma` — держать как compatibility/dev-only. Он полезен для локальных сценариев и совпадает с дефолтами `Open WebUI`, но для нашего административного контура и серверной эксплуатации не должен быть главным ориентиром.

### Лучшие практики по модульности
- Не связывать выбор `vector_store` с выбором `extractor` или `embedder`. Это отдельные слои, которые должны комбинироваться декларативно.
- Строить пайплайн как набор независимых стадий: `extract -> normalize -> chunk -> embed -> index -> verify`. Тогда смена эмбеддера не требует повторного OCR, а смена OCR не ломает правила индексации.
- Для извлечения документов нужен единый реестр провайдеров. На первом этапе стоит поддерживать `tika`, `docling`, `document_intelligence`, `mistral_ocr`, `mineru`, `external`.
- `Docling` нужно рассматривать как основной извлекатель для сложных PDF, офисных документов и таблиц. Он умеет настраиваемый `OCR`, а в `Open WebUI` уже ожидается как один из штатных движков.
- `Tika` нужен как широкий и дешёвый fallback-движок для форматов, где важнее покрытие, чем сложная структурная разметка. Для OCR он опирается на `Tesseract`.
- Отдельный слой параметров `OCR` обязателен. В частности, `Docling` уже поддерживает `tesseract`, `easyocr`, `rapidocr`, `ocrmac`, и эта настройка не должна быть зашита в код конкретной коллекции.
- Нужна сущность профиля коллекции: один сохранённый набор настроек `extractor`, `ocr`, `chunk_size`, `chunk_overlap`, `embedder`, `reranker`, `vector_store`, который можно повторно применять и версионировать.

### Лучшие практики по коллекциям и индексам
- Для `Qdrant` не надо плодить сотни и тысячи коллекций. Документация рекомендует в большинстве сценариев держать одну физическую коллекцию на модель эмбеддингов и разделять данные через `payload`-поля и фильтры.
- Для `Qdrant` обязательно индексировать наиболее селективные `payload`-поля, по которым реально будут идти фильтры. Для нас это минимум `knowledge_base_id`, `file_id`, `hash`, `tenant_id` или `workspace_id`, при необходимости `doc_type`.
- Для `Qdrant` нужно закладывать `aliases` как штатный механизм перевыпуска индекса при смене эмбеддера или рецепта чанкинга.
- Для `pgvector` нужно с самого начала предусмотреть индексы не только на вектор, но и на поля фильтрации. Документация прямо рекомендует обычные индексы по фильтрующим колонкам, а также `partial indexes`, `partitioning` и повышение `hnsw.ef_search` или `iterative scans`, когда фильтрация режет выдачу.
- Для `pgvector` нельзя опираться только на один `document_chunk` без дополнительной метамодели. Нужны явные сущности логической коллекции, версии индекса и активной проекции.
- Для `Milvus` и `Weaviate` многоарендность надо включать только там, где она действительно нужна. В `Milvus` есть несколько стратегий с разными компромиссами, а в `Weaviate` tenant обязан явно проходить через операции поиска и записи.
- Гибридный поиск dense+lexical нужно считать базовым режимом для крупных документных корпусов. И `Qdrant`, и `Milvus`, и `Weaviate`, и `Open WebUI` уже поддерживают этот режим как нормальную практику, а не как экзотику.

### Совместимость с `Open WebUI`
- `Open WebUI` опирается не только на векторную коллекцию, но и на отдельную реляционную сущность `knowledge`, связи `knowledge_file` и служебную коллекцию `knowledge-bases` для поиска по самим базам знаний.
- Для содержимого `knowledge base` `Open WebUI` использует коллекцию по `knowledge.id`, а для одиночных файлов встречается соглашение `file-{file_id}`.
- В метаданных чанков `Open WebUI` уже использует как минимум `file_id`, `name`, `source`, `hash` и `embedding_config`. Если мы хотим потом отображать те же данные в `Open WebUI`, надо не расходиться с этим минимумом.
- Для привязки к модели `Open WebUI` считает основным полем `model.meta.knowledge`, но всё ещё умеет читать старые `collection_name` и `collection_names`. Значит, наш экспортный слой должен писать новый формат и уметь читать старый.
- `Open WebUI` различает `Focused Retrieval` и `Full Context`. В нашей схеме коллекции нужен явный режим выдачи, а не только технические параметры индекса.
- В `Open WebUI` смена `embedding model` или чанкинга требует `reindex` базы знаний. Это надо перенести в план как обязательную операцию продукта, а не как ручную договорённость.

### Рекомендуемая внутренняя схема форка
- Держать три уровня сущностей: `knowledge_base` как логическая сущность UI, `document` как единица загруженного источника, `index_projection` как физическая проекция под конкретный `vector_store`, `embedder` и рецепт чанкинга.
- На уровне продукта явно развести `training dataset` и `RAG dataset`. Первый служит обучению, второй служит индексации и поиску; один может быть производным от другого, но хранить, версионировать и отображать их нужно как разные сущности.
- Хранить отдельно извлечённый и нормализованный текст документа. Это позволит заново считать чанки и эмбеддинги без повторного OCR.
- Для каждого чанка хранить минимум: `knowledge_base_id`, `document_id`, `file_id`, `hash`, `name`, `source`, `content_type`, `created_by`, `extractor`, `ocr_engine`, `chunk_index`, `start_index`, `page`, `section_path`, `language`, `embedding_config.engine`, `embedding_config.model`, `chunk_recipe`.
- В UI показывать логические коллекции, а не физические индексы. Пользователь должен видеть одну коллекцию, даже если под ней уже лежит несколько версий индекса.
- Для `Open WebUI` сделать отдельную проекцию или синхронизатор, который умеет материализовать `knowledge`, `knowledge_file`, служебную коллекцию `knowledge-bases` и рабочие коллекции содержимого.

### Текущий графовый интерфейс и контур `RAG UI`
- В текущем `unsloth` уже есть полноценная доска на `@xyflow/react` внутри `recipe-studio`. Там присутствуют узлы, семантические связи, боковая панель настройки блока, плавающие контролы запуска и проверки, а также отдельный просмотр выполнения. Значит, для сложного `RAG`-ingestion у продукта уже есть подходящий визуальный паттерн.
- Из этого не следует, что весь `RAG` надо делать только через доску. Правильнее держать два уровня интерфейса: быстрый путь через карточки и формы для типовых действий и расширенный `board` для сложных конвейеров и экспериментальных профилей ingestion.
- Быстрый путь должен закрывать основной сценарий администратора: выбрать `RAG`-бэкенд, выбрать или создать коллекцию, выбрать профиль ingestion, загрузить `RAG dataset`, запустить индексацию, увидеть статус задания, при необходимости выполнить `reindex` и синхронизацию с `Open WebUI`.
- Расширенный `board` нужен там, где пользователь хочет явно собрать конвейер из стадий `extract -> normalize -> OCR -> chunk -> embed -> index -> sync`, сравнить два провайдера, вставить проверку качества, посмотреть промежуточный текст и повторно прогнать отдельный этап.
- Карточки на обзорных страницах должны быть компактными и операционными, а не декоративными. Для коллекций нужны поля: имя, `backend`, активный индекс, профиль ingestion, количество документов, объём чанков, эмбеддер, статус синхронизации, время последнего задания. Для заданий нужны: тип, коллекция, инициатор, текущий этап, прогресс, журнал, предупреждения, итоговый статус.
- Визуально новый `RAG UI` должен повторять текущий язык `Studio`: спокойная светлая поверхность, тонкие границы, локальные статусные акценты, небольшие заголовки и минимальный шум. Повторно использовать существующие узлы и оболочки важнее, чем придумывать новый вид карточек.
- Сам `board` не должен быть копией графа для генерации датасетов. Семантика узлов должна быть другой: источник корпуса, извлечение, OCR, нормализация, чанкинг, эмбеддинг, индекс, синхронизация, валидация поиска. Но механика размещения, связи, запусков и настройки блока может быть общей.

### Выводы из `agent-navigator-pro`
- Самый полезный паттерн оттуда — не прямой `Qdrant` внутри workflow, а граница `KnowledgeBaseStoreProtocol` плюс конкретная реализация `QdrantKnowledgeBaseStore`. Это хороший ориентир для нашего форка: `execution` и `retrieval` должны зависеть от абстракции хранилища.
- Там уже зафиксировано разделение `session_rag` и `knowledge_base_rag` без дублирования всей логики поиска. Правильный смысл не в двух отдельных `RAG`-системах, а в одном backend retrieval layer с разным `scope policy`.
- В `agent-navigator-pro` для `Qdrant` явно закладываются поля `collection_id`, `source_id`, `document_id`, `document_version_id`, `thread_id`, `source_scope`, `expires_at`, `embedding_model_id`, `index_version`, `chunking_version`. Это хороший референс для нашей будущей схемы payload и metadata.
- Полезная идея оттуда: session-документы могут жить в том же `Qdrant`, что и knowledge base, но как короткоживущая область с `TTL`-политикой и фильтрами, а не как полностью отдельный backend.
- Ещё один полезный паттерн — persisted chunk embeddings scaffold. Это даёт возможность не переэмбеддить уже сохранённые KB chunks на каждом запросе и использовать query-time embedding только для самого запроса.
- В `agent-navigator-pro` уже есть deterministic merge policy между `knowledge` и `session`: per-scope candidate budgets, dedup по нормализованному тексту, предпочтение `session overlay` при близких оценках. Этот паттерн стоит перенести в требования, даже если точная формула потом будет другой.
- Операторская часть там тоже ценна: `Open WebUI` и backend `session RAG` могут использовать один сервер `Qdrant`, но обязаны быть разведены по namespace и owner semantics. Это особенно важно, если мы потом будем отображать наши коллекции в `Open WebUI`, но не хотим смешать native `Knowledge` и наш административный контур.

## External references
- `Open WebUI RAG`: https://docs.openwebui.com/features/rag
- `Open WebUI Knowledge`: https://docs.openwebui.com/features/workspace/knowledge/
- `Open WebUI Document Extraction`: https://docs.openwebui.com/features/chat-conversations/rag/document-extraction/
- `Open WebUI Docling`: https://docs.openwebui.com/features/rag/document-extraction/docling/
- `Qdrant Collections`: https://qdrant.tech/documentation/concepts/collections/
- `Qdrant Payload`: https://qdrant.tech/documentation/concepts/payload/
- `Qdrant Multitenancy`: https://qdrant.tech/documentation/guides/multitenancy/
- `Qdrant Hybrid Queries`: https://qdrant.tech/documentation/concepts/hybrid-queries/
- `pgvector`: https://github.com/pgvector/pgvector
- `Milvus Filtered Search`: https://milvus.io/docs/filtered-search.md
- `Milvus Multi-tenancy`: https://milvus.io/docs/multi_tenancy.md
- `Weaviate Hybrid Search`: https://docs.weaviate.io/weaviate/concepts/search/hybrid-search
- `Weaviate Multi-tenancy`: https://docs.weaviate.io/weaviate/manage-data/multi-tenancy
- `Weaviate Collection Aliases`: https://docs.weaviate.io/weaviate/manage-collections/collection-aliases
- `Chroma Architecture`: https://docs.trychroma.com/docs/overview/architecture
- `Chroma Manage Collections`: https://docs.trychroma.com/docs/collections/manage-collections
- `Chroma Metadata Filtering`: https://docs.trychroma.com/docs/querying-collections/metadata-filtering
- `Docling Installation`: https://docling-project.github.io/docling/getting_started/installation/
- `Docling Pipeline Options`: https://docling-project.github.io/docling/reference/pipeline_options/
- `Apache Tika Getting Started`: https://tika.apache.org/2.1.0/gettingstarted.html
- `Apache Tika OCR Parser`: https://tika.apache.org/2.9.4/api/org/apache/tika/parser/ocr/TesseractOCRParser

## Open questions
- Как назовём новый административный маршрут: отдельный `/rag-admin`, `/admin/rag` или расширение существующего `data-recipes`?
- Хотим ли мы вести `RAG Upload` как часть `Data Recipes`, или это должен быть полностью отдельный контур поверх `FastAPI` и `Qdrant`?
- Делаем ли расширенный `RAG board` сразу в первом этапе, или сначала закрываем типовой поток карточками и формами, а доску добавляем вторым шагом поверх тех же backend-сущностей?
- Нужен ли нам сразу `docker-compose.yml` с `Qdrant` и вспомогательными сервисами, или на первом этапе достаточно одного контейнера `Studio`?
