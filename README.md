# deteccao_de_anomalias_em_transacoes.py
**Atividade proposta num bootcamp da DIO para machine learning de detecção de anomalias em transações de cartões de crédito, testando diferentes balanceamentos de dados, modelos de machine learning e suas configurações para sensibilidade aos dados enquanto mantém boa recall e precisão.**

*No curso, a base do código foi entregue e a proposta era que o otimizássemos para melhor eficiência ao que é proposto. Assim devíamos realizar diversos testes com modelos e métodos de balanceamento de dados para encontrar as melhores soluções.* 

*Minha solução para definir se o aprendizado do modelo estava de fato eficiente e sua reprodutibilidade garantiria uma média decente de eficiência, utilizei StratifiedKFold e validação cruzada para analisar diversos resultados de recall de diferentes folds. A cada tentativa, 20 folds eram testados, e uma média foi gerada entre eles, junto de seu desvio padrão. Também usei classification report e matriz de confusão para analisar melhor a precisão e F1 score para ter mais clareza sobre a eficiência do modelo.* 

*O modelo utilizado foi XGBoost, que combinado com ADASYN pude obter os melhores resultados, tanto em precisão e recall quanto em estabilidade nos testes em diferentes folds.* 


**Resultados obtidos no último teste com as configs do arquivo fraudes1.py:**
--------------------------------------------
**Recall de cada fold:**   
`0.8        0.84       0.84       0.84       0.88`       
`0.88       0.84       0.95833333 0.79166667 0.91666667` 

`0.79166667 0.83333333 0.83333333 0.95833333 0.875`      
`0.8        0.84       0.76       0.92       0.8`
 
**Recall médio:**  `0.8499166666666668`

**Desvio padrão do recall:**  `0.05417685801561812`

**Classification report:**  precision    recall  f1-score   support

                   0       1.00      1.00      1.00     85295
                   1       0.94      0.80      0.86       148

            accuracy                           1.00     85443
           macro avg       0.97      0.90      0.93     85443
        weighted avg       1.00      1.00      1.00     85443

**Matrix de confusão** (resultados de true positives que passaram despercebidos nas fraudes não me agradou, porém, a meta principal era atingir a melhor estabilidade e eficiência no recall enquanto mantinha uma boa precisão: 

    [85287     8]
    [   30   118]

----------------------------------
Está faltando neste repositório: 
- Análise sobre variáveis que mais influenciaram o machine learning.
- Ajuste e teste de hiperparâmetros, testes finais para conclusão definitiva.
