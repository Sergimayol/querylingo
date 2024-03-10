----------- text_generation -----------
CREATE TABLE dataset_text_generation AS
SELECT
    tg."index",
    tg.question || ' ' || tg.context || ' ' || tg.answer AS text
FROM
    raw_text_generation tg;

----------- text_completition -----------
CREATE TABLE dataset_text_completition AS
SELECT
    tg."index",
    '### Instruction: ' || tg.question || ' ### Input: ' || REPLACE(tg.context, '"', '''') || ' ### Response: ' || REPLACE(tg.answer, '"', '''') AS text
FROM
    raw_text_generation tg;

INSERT INTO
    dataset_text_completition
SELECT
    i."index",
    i."input" || i."output"
FROM
    raw_instructs i
WHERE
    i."source" IN (
        'Llama-2-SQL-Dataset-eval-00000-of-00001-6907aec719559d7d.jsonl',
        'Llama-2-SQL-Dataset-train-00000-of-00001-922416e34c5bc71c.jsonl',
        'Llama-2-SQL-Dataset-val-00000-of-00001-98c87bd893ed1bdb.jsonl'
    );

----------- chatbot -----------
--- Model: TinyLlama/TinyLlama-1.1B-Chat-v0.1
CREATE TABLE dataset_chatbot AS
SELECT
    tg."index",
    '### Human: ' || tg.question || ' ' || REPLACE(tg.context, '"', '''') || '### Assistant: ' || REPLACE(tg.answer, '"', '''') AS text
FROM
    raw_text_generation tg;

----------- chatbot_instruct -----------
--- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
CREATE TABLE dataset_chatbot_instruct AS
SELECT
    tg."index",
    '<|system|> You are a chatbot who can help making SQL queries!</s> <|user|> ' || tg.question || ' ' || REPLACE(tg.context, '"', '''') || '</s> <|assistant|>' || REPLACE(tg.answer, '"', '''') AS text
FROM
    raw_text_generation tg;

----------- text_instruct -----------
--- Model: mistralai/Mistral-7B-Instruct-v0.2
CREATE TABLE dataset_text_instruct AS
SELECT
    tg."index",
    '<s> [INST] ' || tg.question || ' The context is: ' || REPLACE(tg.context, '"', '''') || ' [/INST] ' || REPLACE(tg.answer, '"', '''') || '</s>' AS text
FROM
    raw_text_generation tg;

INSERT INTO
    dataset_text_instruct
SELECT
    i."index",
    i."text"
FROM
    raw_instructs i
WHERE
    i."source" = 'llama2-sql-instruct-train.jsonl';

INSERT INTO
    dataset_text_instruct
SELECT
    i."index",
    '<s> ' || i."input" || ' ' || i."output" || '</s>'
FROM
    raw_instructs i
WHERE
    i."source" IN (
        'spider_text_to_sql-train-00000-of-00001-36a24700f19484dc.jsonl',
        'spider_text_to_sql-validation-00000-of-00001-fa01d04c056ac579.jsonl'
    );

INSERT INTO
    dataset_text_instruct
SELECT
    i."index",
    '<s> ' || i."text" || '</s>'
FROM
    raw_instructs i
WHERE
    i."source" = 'sql-create-context-instruction-train-00000-of-00001-ea1a61c2db38e8fc.jsonl';

----------- translation -----------
--- Model: google/t5-*
CREATE TABLE dataset_translation AS
SELECT
    tg."index",
    'Translate to SQL: ' || tg.question || ' ' || REPLACE(tg.context, '"', '''') || ' SQL: ' || REPLACE(tg.answer, '"', '''') AS text
FROM
    raw_text_generation tg;

----------- sys_prompts -----------
--- Model: llama2-sql-instruct-sys-prompt
CREATE TABLE dataset_sys_prompt AS
SELECT
    lsispt."text"
FROM
    "llama2-sql-instruct-sys-prompt-train" lsispt;