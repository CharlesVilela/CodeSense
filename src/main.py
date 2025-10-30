from entity.DataIngestion import DataIngestion
from entity.RAGDataProcessor import RAGDataProcessor
from entity.Diagnose_RAG import diagnose_rag_issues, fix_rag_system, create_sample_queries_based_on_content

def execute():

    data_ingestion = DataIngestion()
    rag = RAGDataProcessor()
    while True:
        input_user = input('Digite a opção: [1] - Coletar Documentos Oficiais [2] - Coletar no GitHub [3] - Gerar o RAG das coletas [4] - Demonstrar o uso do RAG [5] - Diagnostico do RAG [0] - SAIR: ')
        if input_user == '1':
            print('Coletar Documentos Oficiais')
            data_ingestion.fetch_technical_docs()
        elif input_user == '2':
            print('Coletar no GitHub')
            data_ingestion.extract_github_content()
        elif input_user == '3':
            print('Gerar o RAG')
            rag.process_all_data()
        elif input_user == '4':
            print('Demonstrar o uso do RAG')
            data_ingestion.demonstrate_rag_usage()
        elif input_user == '5':
            print("🚀 INICIANDO DIAGNÓSTICO COMPLETO DO RAG")
            print("=" * 50)

            # Fase 1: Diagnóstico
            has_data = diagnose_rag_issues()

            if has_data:
                # Fase 2: Correções
                fix_rag_system()

                # Fase 3: Consultas baseadas no conteúdo
                create_sample_queries_based_on_content()
            else:
                print("\n❌ Sistema RAG não tem dados suficientes.")
                print("💡 Execute primeiro a coleta de dados:")
                print("   - python seu_script_de_coleta.py")
        elif input_user == '0':
            print('Sair')
            break


# Exemplo de uso específico para seu caso
if __name__ == "__main__":
    execute()


# # Executar demonstração
# if __name__ == "__main__":
#     # Primeiro: processar todos os dados
#     processed_chunks = run_full_rag_pipeline()
#
#     # Depois: demonstrar uso
#     demonstrate_rag_usage()

