# Land Use/Land Cover Classification

## Descrição do Projeto

Este projeto visa a **classificação supervisionada de imagens de satélite** para a análise de **detecção de mudanças ambientais** no uso e cobertura da terra (LULC). Utilizando algoritmos de aprendizado de máquina e dados de sensoriamento remoto, o projeto busca mapear diferentes classes de LULC e identificar alterações significativas ao longo do tempo, fornecendo informações cruciais para o monitoramento ambiental e o planejamento territorial.

## Objetivos

- Desenvolver um modelo de classificação supervisionada para mapear classes de uso e cobertura da terra.
- Utilizar imagens de satélite para identificar e quantificar mudanças no LULC.
- Avaliar a precisão do modelo de classificação e a significância das mudanças detectadas.

## Metodologia

1. **Coleta e Pré-processamento de Imagens**: Aquisição de imagens de satélite (ex: Landsat, Sentinel) e aplicação de correções radiométricas e atmosféricas.
2. **Seleção de Amostras de Treinamento**: Identificação manual de áreas de treinamento para cada classe de LULC.
3. **Extração de Atributos**: Cálculo de índices de vegetação, índices de água, texturas e outras características espectrais.
4. **Classificação Supervisionada**: Aplicação de algoritmos de aprendizado de máquina (ex: Random Forest, Support Vector Machine) para classificar as imagens.
5. **Pós-classificação e Detecção de Mudanças**: Refinamento dos resultados e comparação de mapas de LULC de diferentes períodos para identificar mudanças.

## Tecnologias Utilizadas

- **Python**: Linguagem de programação principal.
- **Bibliotecas Python**: `scikit-learn` para algoritmos de aprendizado de máquina, `rasterio` e `gdal` para manipulação de dados raster, `numpy` para computação numérica, `matplotlib` e `seaborn` para visualização.
- **Ferramentas SIG**: QGIS ou ArcGIS para visualização e validação dos resultados.

## Como Usar

1. Clone este repositório:
   ```bash
   git clone https://github.com/tassiaparadasampaio-cmyk/Land-Use-Land-Cover-Classification.git
   ```
2. Instale as dependências necessárias:
   ```bash
   pip install scikit-learn rasterio gdal numpy matplotlib seaborn
   ```
3. Prepare seus dados de imagem de satélite e amostras de treinamento.
4. Execute o script principal (exemplo):
   ```bash
   python lulc_classification.py
   ```

## Exemplo de Código (lulc_classification.py)

```python
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

print("Script de exemplo para classificação de uso e cobertura da terra.")

# Este é um placeholder. O código real exigiria dados de imagem de satélite e amostras de treinamento.
# Para fins de demonstração, vamos criar dados sintéticos.

# Dados sintéticos: 3 bandas (Red, Green, Blue) e 2 classes (Floresta, Água)
# Imagine uma imagem de 100x100 pixels
num_pixels = 10000
num_bands = 3

# Gerar dados aleatórios para as bandas
X = np.random.rand(num_pixels, num_bands) * 255 # Valores de 0 a 255

# Gerar classes de LULC sintéticas (0 para Floresta, 1 para Água)
y = np.random.randint(0, 2, num_pixels) 

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar e treinar um classificador Random Forest
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred = classifier.predict(X_test)

# Avaliar o modelo
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Exemplo de como visualizar uma parte dos dados (apenas para demonstração)
# Não é uma visualização de mapa real, mas mostra a ideia de classes.
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=\'viridis\', s=10, alpha=0.7)
plt.title(\'Classificação LULC Sintética (Exemplo)\')
plt.xlabel(\'Banda 1 (Red)\')
plt.ylabel(\'Banda 2 (Green)\')
plt.colorbar(ticks=[0, 1], label=\'Classe (0: Floresta, 1: Água)\')
plt.show()

print("\nPara uma aplicação real, você precisaria de:")
print("- Imagens de satélite reais (ex: .tif)")
print("- Vetores de amostras de treinamento (ex: .shp ou .geojson)")
print("- Funções para extrair valores das bandas e atributos para as amostras.")
```

## Contribuição

Sinta-se à vontade para contribuir com este projeto. Por favor, abra uma issue ou envie um pull request.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
