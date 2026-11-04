# Detecção de fraudes em cartão de crédito

Este projeto treina classificadores para identificar fraudes (classe `1`) em transações de cartão. A fraude representa cerca de 0,17% dos registros: prever sempre “normal” ainda daria aproximadamente 99,83% de acurácia, mas recall zero para fraude. Por isso, a comparação prioriza recall, junto com precisão e F1 da classe positiva.

## Preparação e abordagem

- O notebook carrega o CSV público diretamente do link; o dataset não fica no repositório.
- A base tem 284.807 linhas e 31 colunas: `Time`, `Amount`, `Class` e `V1`–`V28`; as últimas foram transformadas por PCA.
- Cria `Amount_log` com `numpy.log1p`.
- Separa treino, validação e teste de forma estratificada. O scaler é ajustado no treino; a validação escolhe o limiar; o teste final mede generalização.
- Compara regressão logística, Random Forest com pesos de classe e XGBoost com `scale_pos_weight` calculado no treino.
- Compara regressão logística com treino original, undersampling e SMOTE, sempre avaliando no mesmo teste original.
- Inclui curvas ROC e precisão-recall, GridSearchCV por recall, importância de variáveis e explicações SHAP global e local.

## Resultados da execução salva

Métricas da classe fraude (`1`) no teste final:

| Modelo/configuração | Precisão | Recall | F1 |
| --- | ---: | ---: | ---: |
| Regressão logística baseline, limiar 0,50 | 0,86 | 0,63 | 0,73 |
| Random Forest, `class_weight="balanced"` | 0,78 | 0,79 | 0,79 |
| XGBoost ponderado, limiar 0,50 | 0,89 | 0,78 | 0,83 |
| Regressão logística, limiar escolhido na validação (0,1147) | 0,75 | 0,72 | 0,73 |
| Regressão logística com `class_weight="balanced"` | 0,07 | 0,88 | 0,12 |

O limiar foi selecionado na validação para maximizar a precisão sujeito a recall mínimo de 0,80. No teste separado, o recall foi 0,716, abaixo da meta: o limiar não garante que a meta se repita em novos dados. A curva ROC da regressão logística teve AUC 0,957. A busca de parâmetros do XGBoost alcançou recall médio 0,844 na validação.

### Comparação de amostragem

Com a mesma regressão logística e limiar 0,50, avaliada no teste original:

| Treino | Precisão fraude | Recall fraude | F1 fraude |
| --- | ---: | ---: | ---: |
| Original | 0,861 | 0,628 | 0,727 |
| Undersampling | 0,045 | 0,885 | 0,085 |
| SMOTE | 0,060 | 0,872 | 0,112 |

Undersampling e SMOTE elevaram o recall, mas produziram muitos falsos alertas, como mostram as precisões baixas. A escolha depende do custo de fraude não detectada versus investigação de transações legítimas.

## O que o SHAP mostrou

No resumo global, `V14` teve a maior magnitude média de contribuição (aprox. 3,10), seguida de `V4` (2,02), `V12` (1,18), `V10` (0,90) e `V11` (0,59). No caso individual exibido, uma fraude real (índice 43204) foi marcada como fraude; `V14` (+6,50), `V4` (+2,63) e `V12` (+2,05) empurraram a saída do modelo na direção da classe fraude, enquanto `V13` (-0,73) contribuiu na direção oposta. Esses valores explicam o modelo, não causas da fraude.

## Diferenças em relação às aulas

O projeto adiciona `Amount_log`, compara três algoritmos, avalia pesos de classe, testa undersampling e SMOTE, escolhe o limiar usando validação e mostra explicações SHAP local e global. O scaler é ajustado somente no treino para evitar vazamento de dados.

## Como reproduzir

Abra `main.ipynb` em Jupyter e execute as células em ordem. O dataset é baixado pelo link que está no notebook. Dependências: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `imbalanced-learn`, `xgboost` e `shap`.
