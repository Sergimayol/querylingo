# Querylingo

Querylingo is a working in process LLM to parse Natural Language Queries to SQL and the other way around. This project is part of my Bachelor Thesis at the University.

## Main Goal

The main goal of this project is to create a tool that can parse natural language queries to SQL and the other way around. This tool should be able to understand the user's query and translate it to a SQL query that can be executed in a database. The tool should also be able to translate a SQL query to a natural language query that can be understood by the user.

## Getting Started

```bash
WORKERS=8 DEBUG=3 python src/data.py -dw -d /mnt/d/tfg/ -e /mnt/d/tfg/processed/hf/ /mnt/d/tfg/processed/ -p -s jsonl -cd /mnt/d/tfg/processed/datasets.sqlite /mnt/d/tfg/processed/datasets/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
