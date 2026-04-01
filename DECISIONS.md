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

## Threshold de decisão
**Escolha inicial:** `0.60`

**Motivo:** Valor de partida conservador, a ser calibrado via curva precision-recall em amostra anotada. Preferimos falsos negativos (no-match) a falsos positivos (match errado), pois dados incorretos no mapeamento de competências têm impacto direto em decisões de negócio.

---

## No-match explícito
**Escolha:** Retornar `no_match` em vez de forçar o melhor candidato quando score < threshold.

**Motivo:** Skills sem correspondência na taxonomia (ex: "espiritualidade", "Alinhamento com o universo") são informação legítima — indicam gaps na taxonomia ou ruído nos dados. Forçar um match mascara esse sinal.

---

## Limitações conhecidas
- Threshold fixo não considera variância por domínio (skills técnicas vs. comportamentais têm distribuições diferentes)
- Sem reranking por LLM: casos ambíguos dependem apenas do embedding
- Não detecta automaticamente quando a taxonomia precisa ser expandida

---

## O que faria com mais tempo
- Calibração de threshold por categoria de skill
- Reranking dos top-3 candidatos com LLM (GPT-4o ou Claude) para casos de score entre 0.55 e 0.75
- Pipeline de avaliação contínua com synthetic ground truth (paráfrases automáticas)
- Interface de revisão humana para casos de baixa confiança
