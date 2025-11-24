import nbformat
import os

def load_reference_cells():
    """
    Lê o notebook e retorna uma LISTA de strings (código limpo).
    Remove linhas que são apenas comentários (#) para economizar tokens.
    """
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(base_path, "knowledge_base", "Funcoes_Credito_Teste.ipynb")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Notebook não encontrado: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = []
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = cell.source
            
            cleaned_lines = []
            for line in source.splitlines():
                stripped_line = line.strip()
                
                if stripped_line and not stripped_line.startswith("#"):
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                code_cells.append("\n".join(cleaned_lines))
    
    return code_cells