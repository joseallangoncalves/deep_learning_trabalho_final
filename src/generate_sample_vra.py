"""
Script de Geração de Amostra Sintética Estrutural do VRA (ANAC)
Permite validar e testar todo o pipeline de ponta a ponta sem dependência imediata de download externo.
"""
import os
import numpy as np
import pandas as pd

def generate_sample_vra_dataset(
    output_path: str = "data/raw/vra_amostra_2023.csv",
    n_samples: int = 5000,
    random_seed: int = 42
) -> pd.DataFrame:
    np.random.seed(random_seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    airlines = ['AZU', 'GLO', 'TAM', 'PTB', 'ONE', 'PAM']
    airports = ['SBBR', 'SBGR', 'SBSP', 'SBCF', 'SBRJ', 'SBKP', 'SBCG', 'SBPA', 'SBFZ', 'SBSV']
    tipos_linha = ['N', 'N', 'N', 'I', 'R']
    situacoes = ['REALIZADO', 'REALIZADO', 'REALIZADO', 'REALIZADO', 'CANCELADO']
    
    # Geração de timestamps cronológicos em 2023
    base_dates = pd.date_range(start='2023-01-01 06:00', end='2023-03-31 23:00', periods=n_samples)
    
    records = []
    for dt in base_dates:
        empresa = np.random.choice(airlines, p=[0.35, 0.30, 0.25, 0.04, 0.03, 0.03])
        origem = np.random.choice(airports)
        destino = np.random.choice([a for a in airports if a != origem])
        
        # Duração programada de voo (entre 45 min e 240 min)
        duracao_min = np.random.randint(45, 240)
        chegada_prevista = dt + pd.Timedelta(minutes=duracao_min)
        
        situacao = np.random.choice(situacoes, p=[0.24, 0.24, 0.24, 0.24, 0.04])
        
        if situacao == 'CANCELADO':
            partida_real = np.nan
            chegada_real = np.nan
            justificativa = 'METEOROLOGIA ADVERSA'
        else:
            # 80% pontual (atraso entre -15 e +15 min), 20% atraso crítico (> 15 min com cauda longa)
            is_delayed = (np.random.rand() < 0.20)
            if is_delayed:
                atraso_partida = np.random.exponential(scale=35) + 15.0
            else:
                atraso_partida = np.random.normal(loc=0.0, scale=8.0)
                
            atraso_chegada = atraso_partida + np.random.normal(loc=-2.0, scale=5.0)
            
            partida_real = dt + pd.Timedelta(minutes=float(atraso_partida))
            chegada_real = chegada_prevista + pd.Timedelta(minutes=float(atraso_chegada))
            justificativa = 'CONGESTIONAMENTO AEROPORTUÁRIO' if is_delayed else ''
            
        records.append({
            'ICAOEmpresaAerea': empresa,
            'NumeroVoo': str(np.random.randint(1000, 9999)),
            'CodigoDI': '0',
            'CodigoTipoLinha': np.random.choice(tipos_linha),
            'ICAOAerodromoOrigem': origem,
            'ICAOAerodromoDestino': destino,
            'PartidaPrevista': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'PartidaReal': partida_real.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(partida_real) else '',
            'ChegadaPrevista': chegada_prevista.strftime('%Y-%m-%d %H:%M:%S'),
            'ChegadaReal': chegada_real.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(chegada_real) else '',
            'SituacaoVoo': situacao,
            'CodigoJustificativa': justificativa
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_path, sep=';', index=False, encoding='utf-8')
    print(f"[OK] Base de amostra gerada com sucesso em: {output_path} ({len(df)} registros)")
    return df

if __name__ == '__main__':
    generate_sample_vra_dataset()
