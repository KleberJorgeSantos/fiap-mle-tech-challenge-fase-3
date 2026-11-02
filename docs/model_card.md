# Model Card — Classificador de Urgência de Laudos

## Detalhes do modelo

| Item | Valor |
|---|---|
| Tarefa | Classificação de texto em 3 classes: `normal`, `atencao`, `urgente` |
| Arquitetura | `TfidfVectorizer` (uni+bigramas, 50 mil termos, `min_df=2`) + `LogisticRegression` (`C=4`, `class_weight="balanced"`) |
| Formato de serving | ONNX (`model.onnx`, executado com ONNX Runtime); original em `model.joblib` |
| Versionamento | `metadata.json`: versão (timestamp), métricas, paridade ONNX, hash SHA-256 do dataset, hiperparâmetros |
| Treino | `python -m src.pipelines.train` ou a DAG `triage_training_pipeline` (Airflow, semanal) |

## Uso pretendido

- **Apoio à priorização** de laudos em uma fila de leitura: laudos classificados como `urgente`
  sobem para o topo.
- **Não** é um dispositivo de diagnóstico e **não** substitui a avaliação médica. Toda decisão clínica
  continua com o profissional.
- Fora do escopo: textos em outros idiomas (o modelo foi treinado em inglês), notas clínicas muito
  curtas e decisões automáticas sem revisão humana.

## Dados de treino

- **Fonte:** [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
  (Schopf, Braun & Matthes, 2022): 14.438 resumos de artigos médicos em inglês com 5 categorias de doença.
- **Pré-processamento:** normalização ASCII, minúsculas, espaços colapsados, remoção de duplicatas →
  11.227 textos.
- **Rótulos de urgência (proxy):** cardiovascular e sistema nervoso → `urgente`; neoplasias e digestivo
  → `atencao`; condições patológicas gerais → `normal`. Configurável em `params.yaml`.
- **Split:** 80/20 estratificado (`random_state=42`): 8.981 treino / 2.246 teste.

## Métricas (conjunto de teste)

| Métrica | Valor |
|---|---|
| Acurácia | 0,729 |
| **Macro-F1** | **0,727** |
| F1 `normal` | 0,622 |
| F1 `atencao` | 0,792 |
| F1 `urgente` | 0,768 |
| **Recall `urgente`** | **0,753** |
| Paridade de rótulos sklearn × ONNX | 100% |

**Quality gate** (bloqueia publicação): macro-F1 ≥ 0,60 e paridade ONNX ≥ 99%.

## Limitações conhecidas

1. **Rótulos proxy.** Urgência foi derivada da *categoria de doença*, não de uma avaliação clínica real.
   Um laudo cardiológico de rotina vira `urgente` e uma sepse descrita em "condições gerais" vira
   `normal`. Em produção real, o modelo deve ser re-treinado com rótulos de triagem do próprio hospital.
2. **Domínio.** Resumos de artigos científicos têm linguagem diferente de laudos de exame.
3. **Idioma.** Treinado em inglês; laudos em português exigem novo corpus.
4. **Classe `normal`** é a mais difícil (F1 0,62): a categoria original é heterogênea.
5. **Recall de `urgente` de 75%** significa que 1 em cada 4 laudos urgentes não é priorizado. O sistema
   deve **complementar**, nunca substituir, a fila atual.
6. **`sublinear_tf` desligado** para garantir paridade numérica no ONNX (custo de ~0,5 p.p. de macro-F1).

## Monitoramento em produção

- Distribuição das classes previstas (`predictions_total`) como indicador de *drift* de entrada.
- Latência de inferência por backend (`model_inference_seconds`) e latência HTTP p50/p95/p99.
- Taxa de erro (4xx/5xx).
- Retreino semanal pela DAG, sempre passando pelo quality gate antes de publicar.

## Considerações éticas e LGPD

- Laudos são **dados sensíveis de saúde** (LGPD, art. 11). A API não persiste textos e não registra
  conteúdo em logs.
- Possíveis vieses do corpus (publicações científicas de países desenvolvidos) podem não refletir a
  população atendida.
- Recomenda-se ajustar o limiar de `urgente` para priorizar **recall** (custo de um falso negativo é
  maior que o de um falso positivo) e auditar periodicamente os erros com a equipe clínica.

## Reprodução

```bash
uv sync --all-extras
uv run python -m src.pipelines.train      # treina, avalia, exporta e publica em models/
uv run python -m scripts.benchmark        # latência sklearn × ONNX
```
