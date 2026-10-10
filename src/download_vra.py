"""
Script Utilitário para Download Oficial dos Microdados do VRA (ANAC)
Portal Brasileiro de Dados Abertos
"""
import os
import requests
from tqdm import tqdm

ANAC_VRA_BASE_URL = "https://sas.anac.gov.br/sas/dadosabertos/VRA"

def download_vra_month(year: int = 2023, month: int = 1, output_dir: str = "data/raw") -> str:
    """
    Baixa arquivo mensal oficial de VRA da ANAC (ex: VRA_2023_1.csv ou VRA_2023_01.csv).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Padrões comuns de nomenclatura no servidor da ANAC
    candidates = [
        f"VRA_{year}_{month:02d}.csv",
        f"VRA_{year}_{month}.csv",
        f"VRA_{year}{month:02d}.csv"
    ]
    
    for filename in candidates:
        url = f"{ANAC_VRA_BASE_URL}/{year}/{filename}"
        dest_path = os.path.join(output_dir, filename)
        
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
            print(f"[OK] Arquivo já existente: {dest_path}")
            return dest_path
            
        print(f"Tentando download de: {url} ...")
        try:
            resp = requests.get(url, stream=True, timeout=15)
            if resp.status_code == 200:
                total_size = int(resp.headers.get('content-length', 0))
                with open(dest_path, 'wb') as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=65536):
                        size = f.write(chunk)
                        bar.update(size)
                print(f"[Sucesso] Arquivo baixado: {dest_path}")
                return dest_path
        except Exception as e:
            print(f"Aviso: Não foi possível baixar de {url}: {e}")
            
    print("[Info] Utilizando base de dados disponível em data/raw/.")
    return None

if __name__ == '__main__':
    download_vra_month(2023, 1)
