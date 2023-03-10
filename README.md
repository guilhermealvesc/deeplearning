# Deep Learning

> Estudos desenvolvidos durante o curso de Deep Learning da Udemy, com código em Python.

## ANNs (Artificial Neural Networks)

### Perceptron

- Modelo de rede neural artificial que possui apenas uma camada de neurônios, ou seja, não possui camadas ocultas.
- O perceptron é composto por um neurônio, que recebe os dados de entrada, processa e envia os dados de saída.
- Redes neurais artificiais com apenas um neurônio são chamadas de perceptrons, e vários perceptrons formam uma rede neural artificial.

### Rede Neural Artificial

- Rede Neural Artificial é um modelo matemático que tenta imitar o funcionamento do cérebro humano, ou seja, é um modelo de aprendizado de máquina que tenta aprender a partir de dados de entrada e saída, e assim, prever novos dados de entrada.
- Possui camadas de entrada, ocultas e de saída, onde cada camada é composta por neurônios, que são responsáveis por receber os dados de entrada, processá-los e enviar os dados de saída para a próxima camada.
- Cada neurônio é uma estrutura composta por pesos, que são responsáveis por multiplicar os dados de entrada e somar os resultados, aplicando depois uma função de ativação que transforma o resultado como um valor entre 0 e 1.
- A função de ativação mais utilizada é a função sigmóide, que é uma função contínua e derivável.

### Descida do Gradiente

- É um algoritmo de otimização que é utilizado para encontrar o mínimo de uma função.
- Utilizado em redes neurais artificias para minimizar o erro da ANN durante o processo de treinamento.
- Pode ser realizado de forma estocástica ou em lote.
  - Estocástica: cálculo do erro para cada dado de entrada (geralmente é um por vez, e gera saídas diferenetes para entradas iguais)
  - Lote: cálculo do erro para um conjunto de dados de entrada (geralmente é o todo)

### Processo de criação de uma ANN

1. Inicialização dos pesos dos neurônios com valores aleatórios próximos a 0
2. Processo de forward-propagation (envio dos dados de entrada para a camada de saída)
3. Cálculo do erro (utiliza base de treino), podendo ser feito na forma de estocástica ou em lote
4. Back-propagation (processo de ajuste dos pesos dos neurônios)
5. Repetir os de 1 a 4 até que o erro seja menor que um valor pré-definido ou que entre em ciclo infinito

### Construção de uma ANN

- O problema: prever se um cliente de um banco irá deixar o banco ou não (churn Modelling).
- Nesse caso, estaremos utilizando uma ANN para classificar, porém pode ser utilizada para regressão (outros problemas).
- A base de dados utilizada é a [Churn Modelling](./ann/Churn_Modelling.csv).
- Para o aprendizado, será utilizado python com a biblioteca tensorflow (ML) e pandas (processamento de dados).
- Todo processo esta descrito no código [aqui](./ann)