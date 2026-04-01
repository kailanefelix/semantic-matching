# Semantic Skill Matching

Solução de mapeamento semântico entre skills brutas de colaboradores e uma taxonomia padronizada, usando sentence embeddings e cosine similarity.

## Problema

Skills são registradas de formas muito diferentes dependendo da empresa, cargo e contexto. `"gestão de pessoas"`, `"People Management"` e `"liderança de times"` podem representar a mesma competência — ou não. Este projeto automatiza esse mapeamento com indicação de confiança e rejeição explícita quando não há match adequado.

## Abordagem

1. Embedar cada skill da taxonomia (nome + descrição) usando `sentence-transformers`
2. Embedar cada skill bruta do input
3. Calcular cosine similarity entre cada par
4. Aplicar threshold para decidir match vs. no-match

## Estrutura
data/
raw/            # CSVs de input (new_skills.csv, skill_taxonomy.csv)
output/         # Resultado do mapeamento
notebooks/        # Exploração e experimentos
src/              # Código modular de produção
requirements.txt

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso
```bash
python src/match.py --skills data/raw/new_skills.csv --taxonomy data/raw/skill_taxonomy.csv --output data/output/result.csv
```

## Output

O CSV de saída contém:

| Campo | Descrição |
|---|---|
| `skill_raw` | Skill original do input |
| `taxonomy_id` | ID da skill na taxonomia (vazio se no-match) |
| `taxonomy_name` | Nome padronizado |
| `score` | Cosine similarity (0 a 1) |
| `match_status` | `matched` ou `no_match` |

## Decisões técnicas

Documentadas em `DECISIONS.md`.

---