# deteccao_de_anomalias_em_transacoes.py
Atividade proposta num bootcamp da DIO, onde devia otimizar um código para machine learning de detecção de anomalias em transações de cartões de crédito, testando diferentes balanceamentos de dados, modelos de machine learning e suas configurações para sensibilidade aos dados enquanto mantém boa recall e precisão.

No curso, a base do código foi entregue e a proposta era que otimizássemos o código para melhor eficiência ao que é proposto. Assim devíamos realizar diversos testes com modelos e métodos de balanceamento de dados para encontrar as melhores soluções. 

Minha solução para definir se o aprendizado do modelo estava de fato eficiente e sua reprodutibilidade garantiria uma média decente de eficiência, utilizei StratifiedKFold e validação cruzada para analisar diversos resultados de recall de diferentes folds. A cada tentativa, 20 folds eram testados, e uma média foi gerada entre eles, junto de seu desvio padrão. Também usei classification report e matriz de confusão para analisar melhor a precisão e F1 score para ter mais clareza sobre a eficiência do modelo. 

O modelo utilizado foi XGBoost, que combinado com ADASYN pude obter os melhores resultados, tanto em precisão e recall quanto em estabilidade nos testes em diferentes folds. 
