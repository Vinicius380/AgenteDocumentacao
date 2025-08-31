import os
from langchain_community.document_loaders import TextLoader, PythonLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from langchain_community.document_loaders import TextLoader, PythonLoader, NotebookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI

# === 1. Definir chave da API===
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_INFERENCE_ENDPOINT"] =  ""
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = ""  

# === 2. Inicializar LLM===
llm = AzureOpenAI(  
    api_key=os.environ["AZURE_OPENAI_API_KEY"],  
    azure_endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],  
    api_version="2025-01-01-preview"  
)  
  
response = llm.chat.completions.create(  
    model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],  # modelo=deployment_name  
    messages=[{"role": "user", "content": "Seu prompt aqui"}],  
    temperature=0.4  
)  

def load_documents_from_folder(folder_path):
    """Carrega documentos de uma pasta e suas subpastas"""
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"❌ Pasta '{folder_path}' não encontrada!")
        return documents
    
    supported_extensions = {'.py': PythonLoader, '.ipynb': NotebookLoader, '.txt': TextLoader}
    
    for root, _, files in os.walk(folder_path):  # percorre subpastas também
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_extension = os.path.splitext(file_name)[1]
            
            if file_extension in supported_extensions:
                try:
                    loader_class = supported_extensions[file_extension]
                    loader = loader_class(file_path)
                    doc = loader.load()[0]
                    doc.metadata['source_file'] = os.path.relpath(file_path, folder_path)  # caminho relativo
                    documents.append(doc)
                    print(f"✓ Carregado: {doc.metadata['source_file']}")
                except Exception as e:
                    print(f"✗ Erro ao carregar {file_name}: {e}")
    
    return documents

def analyze_documents_modern(documents, llm):
    """Análise moderna usando invoke() ao invés de run() deprecated"""
    if not documents:
        return "Nenhum documento foi carregado para análise."
    
    # Dividir documentos grandes em chunks menores
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = splitter.split_documents(documents)
    print(f"📄 Total de chunks criados: {len(all_chunks)}")
    
    # Processar em batches para evitar rate limiting
    batch_size = 3
    all_analyses = []
    
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        
        # Criar contexto combinado do batch
        combined_content = "\n\n---ARQUIVO---\n\n".join([
            f"ARQUIVO: {doc.metadata.get('source_file', 'desconhecido')}\n{doc.page_content}" 
            for doc in batch
        ])
        
        prompt = f"""
        Arquivo
        {combined_content}
        Ação:
        """
        
        try:
            # Usar invoke() ao invés de run() deprecated
            response = llm.chat.completions.create(
                model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            # Extrair conteúdo da resposta
            if hasattr(response, 'content'):
                analysis = response.content
            else:
                analysis = str(response)
                
            all_analyses.append(f"=== BATCH {i//batch_size + 1} ===\n{analysis}")
            
            print(f"Processado batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
            
            # Rate limiting - pausa entre requests
            time.sleep(1)
            
        except Exception as e:
            print(f"✗ Erro no batch {i//batch_size + 1}: {e}")
            all_analyses.append(f"=== BATCH {i//batch_size + 1} - ERRO ===\n{str(e)}")
    
    return "\n\n" + "="*80 + "\n\n".join(all_analyses)

def main():
    print("Iniciando análise dos arquivos...")
    
    # === 3. Carregar documentos ===
    folder_path = r""  
    documents = load_documents_from_folder(folder_path)
    
    if not documents:
        print("Nenhum arquivo foi carregado. Verifique se a pasta existe e contém arquivos .py, .ipynb ou .txt")
        return

    print(f"Total de arquivos carregados: {len(documents)}")

    # === 4. Analisar documentos ===
    try:
        analysis = analyze_documents_modern(documents, llm)
        
        # === 5. Exibir resultado ===
        print("\n" + "="*80)
        print("ANÁLISE COMPLETA DOS ARQUIVOS")
        print("="*80)
        print(analysis)
        
        # === 6. Salvar resultado em arquivo ===
        with open("analise_llm.txt", "w", encoding="utf-8") as f:
            f.write("ANÁLISE DOS ARQUIVOS\n")
            f.write("="*80 + "\n\n")
            f.write(analysis)

        print(f"\nAnálise salva em: analise_llm.txt")
        
    except Exception as e:
        print(f"Erro durante análise: {e}")

if __name__ == "__main__":
    main()