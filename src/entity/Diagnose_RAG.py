import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

from src.entity.TechEnglishRAGSystem import TechEnglishRAGSystem
from src.entity.RAGDataProcessor import RAGDataProcessor


def diagnose_rag_issues():
    """
    Diagnóstico completo do sistema RAG
    """
    print("🔍 Iniciando diagnóstico do RAG...")

    BASE_PATH = Path(__file__).resolve().parents[2]
    # 1. Verificar se os chunks existem
    chunks_path = os.path.join(BASE_PATH, 'output', "rag_chunks", "processed_chunks.jsonl")
    if not Path(chunks_path).exists():
        print("❌ Arquivo de chunks não encontrado!")
        return False

    # 2. Carregar e analisar os chunks
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"📊 Total de chunks: {len(chunks)}")

    if len(chunks) == 0:
        print("❌ Nenhum chunk encontrado no arquivo!")
        return False

    # 3. Analisar o conteúdo dos chunks
    total_content_length = 0
    empty_chunks = 0
    sample_contents = []

    for chunk in chunks[:10]:  # Analisar apenas os primeiros 10
        content = chunk.get('content', '')
        total_content_length += len(content)

        if len(content.strip()) < 10:
            empty_chunks += 1
        else:
            sample_contents.append(content[:100])  # Primeiros 100 caracteres

    print(f"📏 Comprimento médio do conteúdo: {total_content_length / len(chunks):.0f} caracteres")
    print(f"⚠️  Chunks vazios: {empty_chunks}")

    # 4. Mostrar exemplos de conteúdo
    print("\n📝 Exemplos de conteúdo (primeiros 100 caracteres):")
    for i, content in enumerate(sample_contents[:3]):
        print(f"   {i + 1}. '{content}...'")

    # 5. Analisar metadados
    technologies = {}
    contexts = {}

    for chunk in chunks:
        tech = chunk.get('metadata', {}).get('technology', 'unknown')
        context = chunk.get('metadata', {}).get('professional_context', 'unknown')

        technologies[tech] = technologies.get(tech, 0) + 1
        contexts[context] = contexts.get(context, 0) + 1

    print(f"\n🔧 Tecnologias encontradas: {technologies}")
    print(f"🎯 Contextos profissionais: {contexts}")

    # 6. Testar o vectorizer com exemplos reais
    print("\n🧪 Testando TF-IDF com consultas reais...")

    # Coletar todo o conteúdo para testar o vectorizer
    all_contents = [chunk.get('content', '') for chunk in chunks if chunk.get('content', '').strip()]

    if len(all_contents) > 0:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(all_contents)
            print(f"✅ Vectorizer treinado com {len(vectorizer.get_feature_names_out())} features")

            # Testar algumas consultas
            test_queries = [
                "bug error problem",
                "function method difference",
                "code review help",
                "api documentation"
            ]

            for query in test_queries:
                query_vec = vectorizer.transform([query])
                similarities = np.dot(query_vec, tfidf_matrix.T).toarray()[0]
                max_similarity = np.max(similarities) if len(similarities) > 0 else 0
                print(f"   '{query}': similaridade máxima = {max_similarity:.4f}")

        except Exception as e:
            print(f"❌ Erro no vectorizer: {e}")
    else:
        print("❌ Nenhum conteúdo válido para treinar o vectorizer")

    return len(chunks) > 0


def fix_rag_system():
    """
    Corrige problemas comuns no sistema RAG
    """
    print("\n🔧 Aplicando correções no sistema RAG...")

    processor = RAGDataProcessor(
        chunk_size=600,  # Chunks menores para melhor precisão
        chunk_overlap=100
    )

    print("🔄 Reprocessando dados...")
    chunks = processor.process_all_data()

    if len(chunks) == 0:
        print("❌ Nenhum dado processado. Verificando fontes...")
        check_data_sources()
        return

    # 2. Salvar chunks atualizados
    processor.save_processed_chunks(chunks)

    # 3. Testar com consultas mais específicas
    test_with_better_queries()


def check_data_sources():
    """
    Verifica se há dados nas fontes originais
    """
    print("\n📂 Verificando fontes de dados...")

    sources = [
        ("technical_docs", "Documentações técnicas"),
        ("github_data", "Dados do GitHub")
    ]

    BASE_PATH = Path(__file__).resolve().parents[2]
    for dir_path, description in sources:
        path = Path(os.path.join(BASE_PATH, 'output',dir_path))
        if path.exists():
            files = list(path.glob("*.json"))
            print(f"✅ {description}: {len(files)} arquivos")

            # Mostrar alguns arquivos
            for file_path in files[:3]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        title = data.get('metadata', {}).get('title', 'Sem título')
                        print(f"   📄 {file_path.name}: {title}")
                except:
                    print(f"   ❌ Erro ao ler {file_path.name}")
        else:
            print(f"❌ {description}: Diretório não existe")


def test_with_better_queries():
    """
    Testa o RAG com consultas mais específicas e técnicas
    """
    print("\n🎯 Testando com consultas mais específicas...")

    rag_system = TechEnglishRAGSystem()

    # Consultas mais específicas baseadas no conteúdo técnico
    technical_queries = [
        # Baseadas em React
        "React component lifecycle",
        "JSX syntax in React",
        "React hooks useState",
        "React props and state",
        # Baseadas em JavaScript/TypeScript
        "JavaScript function parameters",
        "TypeScript interface definition",
        "async await JavaScript",
        "Node.js module exports",
        # Baseadas em Docker
        "Docker container commands",
        "Dockerfile instructions",
        "docker-compose configuration",
        # Baseadas em Git
        "git commit message",
        "branch merge conflict",
        "GitHub pull request",
        # Gerais de programação
        "database connection string",
        "API endpoint definition",
        "error handling try catch",
        "loop iteration array"
    ]

    print("🔍 Executando consultas técnicas...")

    successful_queries = 0
    for query in technical_queries:
        results = rag_system.query_rag(query, n_results=3)

        if results['success'] and results['context_chunks']:
            successful_queries += 1
            print(f"✅ '{query}': {results['total_chunks_found']} resultados")

            # Mostrar o melhor resultado
            best_result = results['context_chunks'][0]
            print(f"   🏆 {best_result['technology']} - {best_result['relevance_score']:.3f}")
        else:
            print(f"❌ '{query}': 0 resultados")

    print(f"\n📈 Estatísticas: {successful_queries}/{len(technical_queries)} consultas com resultados")


def create_sample_queries_based_on_content():
    """
    Cria consultas baseadas no conteúdo real dos chunks
    """
    print("\n🎯 Gerando consultas baseadas no conteúdo real...")

    BASE_PATH = Path(__file__).resolve().parents[2]
    chunks_path = os.path.join(BASE_PATH, 'output', "rag_chunks", "processed_chunks.jsonl")
    if not Path(chunks_path).exists():
        print("❌ Arquivo de chunks não encontrado")
        return

    # Coletar palavras-chave dos chunks
    keywords = set()
    sample_sentences = []

    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            content = chunk.get('content', '')

            # Extrair palavras técnicas comuns
            technical_terms = [
                'function', 'method', 'class', 'object', 'variable',
                'parameter', 'return', 'import', 'export', 'interface',
                'component', 'state', 'props', 'hook', 'container',
                'image', 'build', 'deploy', 'commit', 'merge', 'branch'
            ]

            for term in technical_terms:
                if term in content.lower():
                    keywords.add(term)

            # Coletar sentenças completas como exemplos
            sentences = content.split('.')
            for sentence in sentences:
                if len(sentence.strip()) > 20 and len(sentence.strip()) < 100:
                    sample_sentences.append(sentence.strip())
                    if len(sample_sentences) >= 10:
                        break

    print(f"🔑 Palavras-chave encontradas: {list(keywords)[:10]}...")
    print(f"📝 Sentenças de exemplo:")
    for i, sentence in enumerate(sample_sentences[:3]):
        print(f"   {i + 1}. {sentence}")
