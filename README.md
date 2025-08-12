# Data Product de IA — Raw Layer

Este repositório representa a **camada de entrada (Raw Layer)** de um **produto de dados de IA**, com foco em:
- **Governança computacional**
- **Qualidade e conformidade dos dados**
- **Rastreabilidade ponta a ponta**

O exemplo atual simula um **modelo de propensão de compra**, mas a arquitetura é genérica e pode ser reutilizada para **modelos de machine learning, LLMs ou regras de negócio**.

---

## O que já está implementado

### **Contrato de Dados**
- Definido em YAML (`contracts/input_contract.yaml`)
- Especifica:
  - Schema (tipos, obrigatoriedade, enums, regex, ranges)
  - Regras de unicidade e preenchimento
  - Limites e políticas de qualidade
  - Critérios de aceitação por batch
- Validado automaticamente via **Pandera**

### **Qualidade e Classificação**
- Checagens automáticas: preenchimento, ranges, unicidade, domínios permitidos
- Classificação dos registros em:
  - **Válidos** → seguem para processamento/modelo
  - **Inválidos** → logados para correção/auditoria

### **Logging e Auditoria**
- Registros inválidos armazenados em `logs/`
- Percentual de falhas monitorado
- Máscara automática para PII em logs

### **Catalogação e Metadados**
- Geração de arquivos `.json` e `.csv` com:
  - Status do lote
  - Totais de registros
  - Versão do contrato utilizada
  - Caminhos dos arquivos processados
- Metadados separados para válidos e inválidos

### **Testes Automatizados**
- Teste end-to-end (`tests/test_pipeline_input.py`) que:
  - Gera dataset de exemplo
  - Executa o pipeline
  - Valida saídas (válidos, inválidos, metadados)
- Rodando com `pytest` via Poetry

---

## Estrutura de Pastas

`/ia_conference_mvp/
  raw_layer/
    catalog/
    contracts/
    data/
      catalog/
        metadata_invalid/
        metadata_valid/
      logs/
      processed_raw_data/
      raw_data/
    tests/
`


## Como executar
1. Instale as dependências via Poetry
`poetry install`

2. Insira um arquivo raw_data.csv na pasta data/raw_data/
`user_id,nome,email,cpf,idade,genero,estado_civil,estado,renda_mensal,canal_preferido,frequencia_compras_ult_12m,ticket_medio,dias_desde_ultima_compra,tem_cartao_fidelidade,score_engajamento,comprou_ult_3m
U001,Ana Araujo,anaarauj1@example.org,332.181.960-00,23,F,solteiro,DF,2628.03,site,7,365.53,182,False,0.33,0`

3. Execute o pipeline
`poetry run python pipeline_input.py`

4. Verifique as saídas

    Registros válidos: `raw_layer/data/processed_raw_data/valid/`

    Registros inválidos: `raw_layer/data/logs/`

    Metadados catalogados: `raw_layer/catalog/`

5. Execute os testes
`poetry run pytest -v`

- O teste test_pipeline_input_end_to_end cria um dataset de exemplo, aplica o contrato e verifica se:
- Apenas registros válidos seguem adiante
- Registros inválidos são logados
- Metadados e relatórios são gerados corretamente
