# Decisões Técnicas

## Modelo de Embedding
**Escolha:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)

**Motivo:** O corpus de skills mistura português e inglês (ex: "gestão de pessoas" e "People Management" devem mapear para a mesma skill). Este modelo foi treinado especificamente para similaridade semântica multilingual.

**Trade-off:** Qualidade inferior a modelos maiores (ex: OpenAI text-embedding-3-small), mas sem dependência de API, sem custo e com latência previsível.

---

## O que é embeddado da taxonomia
**Escolha:** Concatenar `skill_name + ": " + description`

**Motivo:** O nome sozinho ("Gestão de Pessoas") é pouco informativo. A descrição fornece contexto semântico rico que melhora o recall para skills expressas de forma indireta.

---

## Normalização do texto de entrada
**Escolha:** Normalização seletiva em `SkillEmbedder.normalize()`:
```python
if text.isupper() and len(text) > 4:
    return text.lower().strip()
return text.strip()
```

**Motivo:** Três estratégias foram comparadas via curva Precision-Recall sobre GT manual (46 pares anotados) e GT sintético:

| Estratégia | Precision @0.45 | Recall @0.45 | F1 @0.45 |
|---|---|---|---|
| `none` (só strip) | 0.976 | 0.870 | 0.919 |
| `full_lower` | 0.951 | 0.848 | 0.897 |
| **`selective`** | **0.976** | **0.891** | **0.932** |

`selective` domina as duas outras: corrige variantes em caixa alta acidental (`PYTHON` → matched, 0.6519) sem quebrar substantivos próprios e siglas curtas (`AWS`, `ETL`, `Git`) que dependem da capitalização para o significado semântico no modelo.

**Regra:** strings completamente em maiúsculo (`isupper()`) **e** com mais de 4 caracteres são normalizadas para minúsculo. Strings mais curtas (siglas como `AWS`, `ETL`, `SQL`) e strings com capitalização mista (`Git`, `PowerBI`) são preservadas.

---

## Threshold de decisão
**Escolha:** `0.45`

**Motivo:** Calibrado via curva Precision-Recall sobre GT manual (notebook 02). Com normalização seletiva e threshold 0.45:
- Precision 0.976, Recall 0.891, F1 0.932 — máximo global dos experimentos.
- O threshold anterior (0.60) era conservador demais: sacrificava recall (0.717) sem ganho expressivo de precision.
- Preferimos falsos negativos (no_match → revisão manual) a falsos positivos (match errado → impacto em decisões de negócio).

---

## No-match explícito
**Escolha:** Retornar `no_match` em vez de forçar o melhor candidato quando score < threshold.

**Motivo:** Skills sem correspondência na taxonomia (`espiritualidade`, `Alinhamento com o universo`, `feedback`) são informação legítima — indicam gaps na taxonomia ou ruído nos dados. Validado nas anotações manuais: essas skills ficam consistentemente abaixo do threshold nas três estratégias de normalização e em toda a faixa 0.30–0.90.

---

## Ground truth para avaliação
**Escolha:** Ground truth manual com 46 pares anotados (+ 4 marcados como sem match real).

**Motivo:** O GT sintético (paráfrases automáticas) superestimava falsos positivos com variações artificiais que o modelo confundia, distorcendo a curva. O GT manual revelou que o modelo é significativamente mais preciso (F1 manual ~0.93 vs F1 sintético ~0.70 no mesmo threshold).

**Multi-label:** Casos ambíguos (`cloud computing` → [66, 67, 68], `data science` → [39, 40]) foram anotados com listas de IDs. O evaluator suporta isso via `_valid_taxonomy_ids()`.

---

## Limitações conhecidas
- `scrum` continua como `no_match` nas três estratégias — gap real na taxonomia (entrada existente é "Metodologias Ágeis", não "Scrum" especificamente)
- Threshold fixo não considera variância por domínio (skills técnicas vs. comportamentais)
- Sem reranking por LLM: casos ambíguos dependem apenas do embedding
- Não detecta automaticamente quando a taxonomia precisa ser expandida

---

## O que faria com mais tempo
- Calibração de threshold por categoria de skill (técnica vs. comportamental)
- Reranking dos top-3 candidatos com LLM para scores em [0.40, 0.55]
- Expandir GT manual com skills de MLOps e LLM frameworks
- Interface de revisão humana para casos de baixa confiança
- Pipeline de avaliação contínua com GT manual incremental
