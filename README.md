# CVRP Benchmark

Este repositório contém várias implementações de solvers para o problema CVRP (Capacitated Vehicle Routing Problem) usadas para benchmark.

Arquivos principais:

- `baseline_solver.py` — setup/infra para o modelo de roteamento.
- `gls_solver.py` — solver usando Guided Local Search (GLS) com OR-Tools.
- `sa_solver.py` — simulated annealing (se presente).
- `ts_solver.py` — tabu search (se presente).
- `main.py` — ponto de entrada para execução/benchmark.
- `data_model.py` — definição dos dados do problema.

Dependências
------------

Instale o OR-Tools (exemplo pip):

```bash
pip install ortools
```

Como usar
---------

1. Abra um terminal na pasta do projeto:

```bash
cd D:\Users\jptin\Desktop\USAL\Codes\CVRPBenchmark
```

2. Execute o exemplo principal:

```bash
python main.py
```

Criar um repositório remoto (GitHub)
-------------------------------

Após o commit inicial local, adicione um remoto e faça push:

```bash
git remote add origin https://github.com/<seu-usuario>/<seu-repo>.git
git push -u origin main
```

Licença
-------

Adicione um arquivo `LICENSE` se quiser publicar com uma licença específica.
