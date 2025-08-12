#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import random
import re

# salva no mesmo diretório do script: raw_layer/data/raw_data/raw_data.csv
OUT_CSV = Path(__file__).resolve().parent / "raw_data.csv"

RNG = np.random.default_rng(42)
random.seed(42)

UFs = ["AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"]
GENEROS = ["F","M","Outro"]
ESTADO_CIVIL = ["solteiro","casado","divorciado","viúvo"]
CANAIS = ["app", "site", "whatsapp", "email"]

def gen_user_id(i:int)->str:
    return f"U{((i-1)%999)+1:03d}"  # garante U001..U999

def gen_nome()->str:
    nomes = ["Ana","Bruno","Carla","Diego","Eva","Felipe","Gustavo","Helena","Igor","Júlia","Kaio","Lívia","Marcos","Nina","Otávio","Paula","Rafa","Sofia","Thiago","Vera","Yasmin","Zeca"]
    sobrenomes = ["Silva","Souza","Oliveira","Santos","Pereira","Rodrigues","Almeida","Mendes","Castro","Gomes","Barbosa","Araujo","Martins","Rocha"]
    prefixos = ["","Dr. ","Sr. ","Sra. "]
    return f"{random.choice(prefixos)}{random.choice(nomes)} {random.choice(sobrenomes)}"

def gen_email(nome:str, i:int)->str:
    base = re.sub(r"[^a-z]", "", nome.lower())
    doms = ["example.com","example.org","example.net"]
    return f"{base[:8]}{i}@{random.choice(doms)}"

def dv_cpf(num: str) -> str:
    soma = sum((10 - i) * int(n) for i, n in enumerate(num[:9]))
    d1 = 11 - (soma % 11); d1 = 0 if d1 >= 10 else d1
    soma = sum((11 - i) * int(n) for i, n in enumerate(num[:9] + str(d1)))
    d2 = 11 - (soma % 11); d2 = 0 if d2 >= 10 else d2
    return f"{num[:3]}.{num[3:6]}.{num[6:9]}-{d1}{d2}"

def gen_cpf_valido()->str:
    base = "".join(str(random.randint(0,9)) for _ in range(9))
    return dv_cpf(base)

def dataset(n:int=500, error_rate:float=0.07)->pd.DataFrame:
    rows = []
    for i in range(1, n+1):
        uid = gen_user_id(i)
        nome = gen_nome()
        email = gen_email(nome, i)
        cpf = gen_cpf_valido()
        idade = int(RNG.integers(18, 75))
        genero = random.choice(GENEROS)
        ec = random.choice(ESTADO_CIVIL)
        uf = random.choice(UFs)
        renda = round(float(max(0.0, RNG.normal(4500, 1800))), 2)
        canal = random.choice(CANAIS)
        freq12 = int(max(0, RNG.poisson(6)))
        ticket = round(float(max(0.0, RNG.normal(260, 120))), 2)
        dias = int(abs(RNG.normal(120, 80)))
        fidel = bool(RNG.choice([True, False]))
        eng = round(float(np.clip(RNG.beta(2, 3), 0, 1)), 2)
        comprou = int(RNG.choice([0,1], p=[0.65, 0.35]))
        rows.append([uid,nome,email,cpf,idade,genero,ec,uf,renda,canal,freq12,ticket,dias,fidel,eng,comprou])

    df = pd.DataFrame(rows, columns=[
        "user_id","nome","email","cpf","idade","genero","estado_civil","estado",
        "renda_mensal","canal_preferido","frequencia_compras_ult_12m","ticket_medio",
        "dias_desde_ultima_compra","tem_cartao_fidelidade","score_engajamento","comprou_ult_3m"
    ])

    # Injeta ~7% de erros distribuídos
    k = int(len(df) * error_rate)
    idxs = RNG.choice(df.index, size=k, replace=False)

    for j, idx in enumerate(idxs):
        tipo = j % 8
        if tipo == 0:
            df.loc[idx, "user_id"] = "XYZ"                     # regex inválida
        elif tipo == 1:
            df.loc[idx, "email"] = np.nan                      # obrigatório ausente
        elif tipo == 2:
            df.loc[idx, "cpf"] = "123.456.789-00"              # CPF inválido
        elif tipo == 3:
            df.loc[idx, "idade"] = 16                          # fora do min
        elif tipo == 4:
            df.loc[idx, "canal_preferido"] = "telefone"        # fora do domínio
        elif tipo == 5:
            df.loc[idx, "score_engajamento"] = 1.5             # fora [0,1]
        elif tipo == 6:
            df.loc[idx, "dias_desde_ultima_compra"] = -5       # negativo
        elif tipo == 7:
            # duplica um email pra testar uniqueness
            dup_src = int(max(1, idx-1))
            df.loc[idx, "email"] = df.loc[dup_src, "email"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Dataset sintético gerado: {OUT_CSV}  (linhas={len(df)}, erros≈{k})")

if __name__ == "__main__":
    dataset()
