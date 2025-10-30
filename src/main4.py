import re
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TeachingQualityFilter:
    """
    Filtro para qualidade pedagógica no ensino de inglês
    """

    def __init__(self):
        self.teaching_patterns = [
            r'how to', r'you can', r'we use', r'this means',
            r'for example', r'in practice', r'typically',
            r'commonly used', r'the purpose of', r'allows you to',
            r'enables', r'provides', r'helps you'
        ]

    def score_teaching_quality(self, text: str) -> int:
        """
        Atribui uma pontuação para qualidade no ensino de inglês
        """
        score = 0

        # +1 ponto para cada padrão pedagógico
        pattern_count = sum(1 for pattern in self.teaching_patterns
                            if re.search(pattern, text.lower()))
        score += pattern_count

        # +2 pontos se explicar conceitos (vs apenas descrever)
        if any(word in text.lower() for word in ['means', 'purpose', 'concept', 'definition']):
            score += 2

        # +1 ponto por diversidade de estruturas gramaticais
        grammar_structures = [
            r'\bcan\b', r'\bshould\b', r'\bif\b', r'\bbecause\b',
            r'\bwhen\b', r'\bwhile\b', r'\bafter\b', r'\bbefore\b'
        ]
        grammar_count = sum(1 for pattern in grammar_structures if re.search(pattern, text))
        score += min(grammar_count, 3)

        # Penalizar conteúdo muito técnico
        technical_density = len(re.findall(r'[{}();=<>&|\-]', text)) / len(text) if text else 0
        if technical_density > 0.1:  # Mais de 10% de caracteres técnicos
            score -= 2

        return max(score, 0)

    def is_pedagogical_content(self, text: str) -> bool:
        """
        Verifica se o conteúdo é bom para ensino de inglês
        """
        if len(text) < 30:
            return False

        # Padrões que indicam conteúdo pedagógico
        teaching_patterns = [
            r'how to', r'you can', r'we use', r'this means',
            r'for example', r'in practice', r'typically',
            r'commonly used', r'the purpose of', r'allows you to',
            r'enables', r'provides', r'helps you'
        ]

        # Verificar se é explicativo (vs apenas descritivo)
        pattern_count = sum(1 for pattern in teaching_patterns
                            if re.search(pattern, text.lower()))

        # Verificar proporção de inglês natural vs técnico
        words = text.split()
        common_english_words = len([w for w in words if len(w) > 2 and w.isalpha()])
        technical_terms = len([w for w in words if '_' in w or w.isupper()])

        return (pattern_count >= 2 and
                common_english_words > technical_terms * 2)

    def add_teaching_context(self, text: str) -> str:
        """
        Adiciona contexto específico para ensino de inglês
        """
        # Identificar padrões gramaticais e vocabulário
        grammar_patterns = {
            'present_simple': r'\b(is|are|does|do|have|has)\b',
            'modals': r'\b(can|could|should|would|may|might)\b',
            'imperative': r'^\s*(Create|Add|Use|Define|Install)\b',
            'passive': r'\b(is|are|was|were) \w+ed\b'
        }

        vocabulary = self.extract_vocabulary_terms(text)
        grammar_notes = self.identify_grammar_points(text, grammar_patterns)

        # Adicionar metadados de ensino
        teaching_header = f"📚 English Learning Focus:\n"
        if vocabulary:
            teaching_header += f"🧠 Vocabulary: {', '.join(vocabulary[:3])}\n"
        if grammar_notes:
            teaching_header += f"📝 Grammar: {', '.join(grammar_notes[:2])}\n"

        return teaching_header + "\n" + text if teaching_header.strip() != "📚 English Learning Focus:\n" else text

    def extract_vocabulary_terms(self, text: str) -> List[str]:
        """Extrai termos que são bons para vocabulário técnico"""
        # Termos compostos úteis para aprendizado
        compound_terms = re.findall(r'\b([a-z]+_[a-z]+|[A-Z][a-z]+[A-Z][a-z]+)\b', text)

        # Verbos técnicos comuns
        tech_verbs = ['define', 'declare', 'initialize', 'iterate', 'execute',
                      'compile', 'debug', 'deploy', 'configure', 'implement']

        found_verbs = [verb for verb in tech_verbs if verb in text.lower()]

        return list(set(compound_terms + found_verbs))

    def identify_grammar_points(self, text: str, grammar_patterns: Dict) -> List[str]:
        """Identifica pontos gramaticais no texto"""
        grammar_points = []

        for point, pattern in grammar_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                grammar_points.append(point)

        return grammar_points

    def filter_by_teaching_score(self, chunks: List[Dict], min_score: int = 3) -> List[Dict]:
        """Filtra chunks baseado na qualidade para ensino"""
        filtered_chunks = []

        for chunk in chunks:
            score = self.score_teaching_quality(chunk['content'])
            if score >= min_score:
                # Adicionar score como metadado e enriquecer conteúdo
                chunk['metadata']['teaching_score'] = score
                enhanced_content = self.add_teaching_context(chunk['content'])
                chunk['content'] = enhanced_content
                filtered_chunks.append(chunk)

        return filtered_chunks


class EnhancedRAGDataProcessor:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.teaching_filter = TeachingQualityFilter()

    def advanced_clean_text(self, text: str) -> str:
        """
        Limpeza avançada que remove HTML, markdown e mantém texto explicativo
        """
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove markdown links e imagens
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)

        # Remove badges e shields
        text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
        text = re.sub(r'!\[.*?\]', '', text)

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove código entre backticks
        text = re.sub(r'`[^`]*`', '', text)

        # Remove caracteres especiais mas mantém pontuação básica
        text = re.sub(r'[^\w\s.,!?;:()\-]', ' ', text)

        # Remove múltiplos espaços
        text = re.sub(r'\s+', ' ', text)

        # Remove linhas que são apenas números ou caracteres especiais
        lines = text.split('.')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Mantém apenas linhas com conteúdo textual significativo
            if (len(line) > 20 and
                    not line.startswith('npm') and
                    not line.startswith('git') and
                    not line.startswith('docker') and
                    not line.startswith('yarn') and
                    re.search(r'[a-zA-Z]{4,}', line)):  # Pelo menos 4 letras consecutivas
                cleaned_lines.append(line)

        text = '. '.join(cleaned_lines)

        return text.strip()

    def extract_explanatory_content(self, content: str) -> str:
        """
        Extrai conteúdo explicativo (inglês) de documentos técnicos
        """
        # Padrões que indicam conteúdo explicativo
        explanatory_patterns = [
            r'(?:This|This document|The|A|An)\s+[a-zA-Z,\s]{10,100}\.',
            r'(?:You can|You should|It is|It\'s)\s+[a-zA-Z,\s]{10,100}\.',
            r'(?:To\s+\w+\s+[a-zA-Z,\s]{10,100}\.)',
            r'(?:For\s+\w+\s+[a-zA-Z,\s]{10,100}\.)',
            r'(?:When\s+[a-zA-Z,\s]{10,100}\.)',
            r'(?:If\s+[a-zA-Z,\s]{10,100}\.)',
        ]

        explanatory_sentences = []
        sentences = re.split(r'[.!?]+', content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 25 or len(sentence) > 500:
                continue

            # Verifica se parece ser inglês explicativo
            has_explanatory_structure = any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in explanatory_patterns
            )

            # Verifica proporção de palavras vs caracteres especiais
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence)
            if len(words) > 5 and len(words) / len(sentence.split()) > 0.6:
                if has_explanatory_structure or len(sentence) > 50:
                    explanatory_sentences.append(sentence)

        return '. '.join(explanatory_sentences)

    def intelligent_chunking_for_english_learning(self, content: str, metadata: Dict) -> List[Dict]:
        """
        Chunking focado em conteúdo para aprendizado de inglês
        """
        # Primeiro: limpeza avançada
        cleaned_content = self.advanced_clean_text(content)

        # Segundo: extração de conteúdo explicativo
        explanatory_content = self.extract_explanatory_content(cleaned_content)

        if not explanatory_content:
            return []

        # Divide por sentenças completas
        sentences = re.split(r'(?<=[.!?])\s+', explanatory_content)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk and len(current_chunk) > 50:
                    chunk_data = self._create_learning_chunk(current_chunk.strip(), metadata)
                    if self._is_valuable_for_learning(chunk_data['content']):
                        chunks.append(chunk_data)
                current_chunk = sentence + " "

        if current_chunk and len(current_chunk) > 50:
            chunk_data = self._create_learning_chunk(current_chunk.strip(), metadata)
            if self._is_valuable_for_learning(chunk_data['content']):
                chunks.append(chunk_data)

        return chunks

    def _is_valuable_for_learning(self, text: str) -> bool:
        """Verifica se o texto é valioso para aprendizado de inglês"""
        if len(text) < 30:
            return False

        # Conta palavras significativas
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        if len(words) < 5:
            return False

        # Verifica se tem estrutura de frase completa
        has_capital = text[0].isupper() if text else False
        has_punctuation = text[-1] in '.!?'

        return has_capital and has_punctuation

    def _create_learning_chunk(self, content: str, metadata: Dict) -> Dict:
        """Cria chunk otimizado para aprendizado"""
        # Metadados enriquecidos para aprendizado
        learning_metadata = {
            'title': metadata.get('title', 'Technical Documentation'),
            'technology': metadata.get('technology', 'unknown'),
            'professional_context': metadata.get('professional_context', 'technical'),
            'english_level': self._estimate_english_level(content),
            'content_type': 'english_learning',
            'source_url': metadata.get('url', ''),
            'word_count': len(content.split()),
            'sentence_count': len(re.findall(r'[.!?]+', content)),
            'vocabulary_complexity': self._calculate_vocabulary_complexity(content)
        }

        return {
            'content': content,
            'metadata': learning_metadata,
            'chunk_id': f"learn_{hash(content) & 0xFFFFFFFF}"
        }

    def _estimate_english_level(self, text: str) -> str:
        """Estima nível de inglês do conteúdo"""
        words = text.lower().split()
        if len(words) < 10:
            return "B1"

        # Palavras complexas comuns em documentação técnica
        complex_words = {
            'implementation', 'configuration', 'optimization', 'architecture',
            'asynchronous', 'concurrent', 'comprehensive', 'documentation',
            'infrastructure', 'deployment', 'synchronization', 'compatibility',
            'functionality', 'repository', 'dependency', 'environment'
        }

        complex_count = sum(1 for word in words if word in complex_words)
        complexity_ratio = complex_count / len(words)

        if complexity_ratio > 0.05:
            return "C1"
        elif complexity_ratio > 0.02:
            return "B2"
        else:
            return "B1"

    def _calculate_vocabulary_complexity(self, text: str) -> float:
        """Calcula complexidade do vocabulário"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        if not words:
            return 0.0

        unique_words = set(words)
        return len(unique_words) / len(words)

    def process_for_english_learning(self, tech_docs_dir: str = "technical_docs",
                                     github_dir: str = "github_data", min_teaching_score: int = 3) -> List[Dict]:
        """
        Processa dados especificamente para aprendizado de inglês
        """
        print("🎯 Processando dados para aprendizado de inglês...")

        # Carrega dados brutos
        documents = self._load_raw_documents(tech_docs_dir, github_dir)

        all_learning_chunks = []

        for doc in documents:
            try:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})

                # Processa para extrair conteúdo de aprendizado
                learning_chunks = self.intelligent_chunking_for_english_learning(content, metadata)
                all_learning_chunks.extend(learning_chunks)

                if learning_chunks:
                    print(f"✅ {metadata.get('title', 'Unknown')}: {len(learning_chunks)} chunks de aprendizado")

            except Exception as e:
                print(f"❌ Erro ao processar {metadata.get('title', 'Unknown')}: {e}")

        # Aplicar filtro de qualidade de ensino
        print("\n🔍 Aplicando filtro pedagógico...")
        filtered_chunks = self.teaching_filter.filter_by_teaching_score(all_learning_chunks, min_score=min_teaching_score)

        print(f"\n📚 Total de chunks para aprendizado: {len(all_learning_chunks)}")
        print(f"🎯 Chunks pedagógicos filtrados: {len(filtered_chunks)}")
        self._analyze_learning_content(filtered_chunks)

        return filtered_chunks

    def _load_raw_documents(self, tech_docs_dir: str, github_dir: str) -> List[Dict]:
        """Carrega documentos brutos"""
        documents = []

        for dir_path in [tech_docs_dir, github_dir]:
            path = Path(dir_path)
            if path.exists():
                for file_path in path.glob("*.json"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc = json.load(f)
                            documents.append(doc)
                    except Exception as e:
                        print(f"❌ Erro ao carregar {file_path}: {e}")

        return documents

    def _analyze_learning_content(self, chunks: List[Dict]):
        """Analisa o conteúdo de aprendizado gerado"""
        if not chunks:
            print("❌ Nenhum chunk de aprendizado gerado")
            return

        english_levels = {}
        technologies = {}
        content_lengths = []
        teaching_scores = []

        for chunk in chunks:
            level = chunk['metadata']['english_level']
            tech = chunk['metadata']['technology']
            length = len(chunk['content'])
            score = chunk['metadata'].get('teaching_score', 0)

            english_levels[level] = english_levels.get(level, 0) + 1
            technologies[tech] = technologies.get(tech, 0) + 1
            content_lengths.append(length)
            teaching_scores.append(score)

        print("\n📊 Análise do Conteúdo de Aprendizado:")
        print(f"🎓 Níveis de Inglês: {english_levels}")
        print(f"🔧 Tecnologias: {technologies}")
        print(f"📏 Comprimento médio: {sum(content_lengths) / len(content_lengths):.0f} caracteres")
        if teaching_scores:
            print(f"🎯 Pontuação de ensino média: {sum(teaching_scores) / len(teaching_scores):.2f}")

        # Mostrar exemplos
        print("\n📝 Exemplos de chunks de aprendizado:")
        for i, chunk in enumerate(chunks[:3]):
            print(
                f"   {i + 1}. [Score: {chunk['metadata'].get('teaching_score', 0)}] [{chunk['metadata']['english_level']}] {chunk['content'][:80]}...")


class EnhancedRAGSystem:
    """
    Sistema RAG otimizado para aprendizado de inglês técnico
    """

    def __init__(self, learning_chunks: List[Dict]):
        self.learning_chunks = learning_chunks
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)  # Inclui trigramas
        )
        self._setup_vector_store()

    def _setup_vector_store(self):
        """Configura o vector store com chunks de aprendizado"""
        if not self.learning_chunks:
            print("❌ Nenhum chunk de aprendizado disponível")
            return

        # Treina o vectorizer
        contents = [chunk['content'] for chunk in self.learning_chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)

        print(f"✅ Vector store configurado com {len(self.learning_chunks)} chunks de aprendizado")

    def query_learning_content(self, question: str, n_results: int = 5) -> Dict:
        """
        Consulta conteúdo para aprendizado de inglês
        """
        if not hasattr(self, 'tfidf_matrix'):
            return {'error': 'Vector store não configurado'}

        try:
            # Transforma a pergunta
            question_vec = self.vectorizer.transform([question])

            # Calcula similaridades
            similarities = cosine_similarity(question_vec, self.tfidf_matrix)[0]

            # Encontra os melhores resultados
            best_indices = np.argsort(similarities)[-n_results:][::-1]

            results = []
            for idx in best_indices:
                if similarities[idx] > 0.1:  # Threshold mínimo de similaridade
                    chunk = self.learning_chunks[idx]
                    results.append({
                        'content': chunk['content'],
                        'technology': chunk['metadata']['technology'],
                        'english_level': chunk['metadata']['english_level'],
                        'relevance_score': float(similarities[idx]),
                        'professional_context': chunk['metadata']['professional_context'],
                        'teaching_score': chunk['metadata'].get('teaching_score', 0)
                    })

            # return {
            #     'question': question,
            #     'results': results,
            #     'total_found': len(results),
            #     'success': True
            # }
            return self.enhance_retrieval_for_learning(question, n_results)

        except Exception as e:
            return {'error': str(e), 'success': False}

    def enhance_retrieval_for_learning(self, question: str, n_results: int = 3) -> Dict:
        """
        Versão SIMPLIFICADA e mais robusta para recuperação
        """
        try:
            # Transforma a pergunta original
            question_vec = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, self.tfidf_matrix)[0]

            # Encontra candidatos com threshold BAIXO para garantir resultados
            candidate_indices = []
            for idx, similarity in enumerate(similarities):
                if similarity > 0.05:  # Threshold BEM mais baixo
                    candidate_indices.append((idx, similarity))

            # Se não encontrou candidatos, pega os top N por similaridade
            if not candidate_indices:
                candidate_indices = [(idx, similarities[idx]) for idx in np.argsort(similarities)[-n_results:][::-1]]

            # Ordena por combinação de relevância e teaching score
            scored_results = []
            for idx, similarity in candidate_indices:
                chunk = self.learning_chunks[idx]
                teaching_score = chunk['metadata'].get('teaching_score', 0)

                # Score mais simples: prioriza teaching score quando a relevância é decente
                if similarity > 0.1:
                    combined_score = similarity + (teaching_score * 0.1)
                else:
                    combined_score = similarity

                scored_results.append({
                    'combined_score': combined_score,
                    'relevance_score': similarity,
                    'teaching_score': teaching_score,
                    'chunk': chunk
                })

            # Ordena pelo score combinado
            scored_results.sort(key=lambda x: x['combined_score'], reverse=True)

            # Prepara resultados finais
            results = []
            for result in scored_results[:n_results]:
                chunk = result['chunk']
                results.append({
                    'content': chunk['content'],
                    'technology': chunk['metadata']['technology'],
                    'english_level': chunk['metadata']['english_level'],
                    'relevance_score': result['relevance_score'],
                    'teaching_score': result['teaching_score'],
                    'combined_score': result['combined_score'],
                    'professional_context': chunk['metadata']['professional_context']
                })

            return {
                'question': question,
                'results': results,
                'total_found': len(results),
                'success': True
            }

        except Exception as e:
            print(f"❌ Erro na recuperação: {e}")
            # Fallback para a busca original
            return self.query_learning_content_fallback(question, n_results)


    def optimize_learning_queries(self, query: str) -> List[str]:
        """
        Expande consultas para melhor recuperação de conteúdo pedagógico
        """
        query_expansions = {
            'how to define a function': [
                "function definition examples",
                "how to create functions in programming",
                "defining methods syntax"
            ],
            'what is the purpose of a variable': [
                "variable usage examples",
                "why use variables in code",
                "variable purpose programming"
            ],
            'explain the concept of': [
                "understanding the concept of",
                "what is the concept of",
                "explanation of"
            ]
        }

        expanded_queries = [query]
        for pattern, expansions in query_expansions.items():
            if pattern in query.lower():
                expanded_queries.extend(expansions)

        return expanded_queries


def demonstrate_enhanced_rag(min_teaching_score=3):
    """
    Demonstra o sistema RAG melhorado para aprendizado de inglês
    """
    print("🚀 INICIANDO SISTEMA RAG APRIMORADO")
    print("=" * 50)

    # 1. Processamento especializado para aprendizado
    processor = EnhancedRAGDataProcessor()
    BASE_PATH = Path(__file__).resolve().parents[1]
    technical_docs = os.path.join(BASE_PATH, 'output', 'technical_docs')
    github_data = os.path.join(BASE_PATH, 'output', 'github_data')
    learning_chunks = processor.process_for_english_learning(technical_docs, github_data,min_teaching_score=min_teaching_score)

    if not learning_chunks:
        print("❌ Não foi possível gerar conteúdo para aprendizado")
        return

    # 2. Sistema RAG especializado
    rag_system = EnhancedRAGSystem(learning_chunks)

    # 3. Consultas otimizadas para aprendizado de inglês técnico
    learning_queries = [
        # Consultas básicas de programação em inglês
        "How to define a function in programming?",
        "What is the purpose of a variable?",
        "Explain the concept of object oriented programming",
        "How do you handle errors in code?",
        "What are the benefits of using version control?",

        # Consultas sobre processos de desenvolvimento
        "How to write good commit messages?",
        "What is code review and why is it important?",
        "How to document your code properly?",
        "What are best practices for software development?",
        "How to work with a development team?",

        # Consultas específicas de tecnologias
        "How to create a React component?",
        "What is Docker used for?",
        "How to use Git for version control?",
        "What are TypeScript interfaces?",
        "How to configure a development environment?",

        # Consultas de comunicação técnica
        "How to explain technical concepts to non-technical people?",
        "What language to use in technical documentation?",
        "How to ask for help with programming problems?",
        "How to describe software architecture?",
        "What vocabulary is used in daily standup meetings?"
    ]

    print("\n🔍 Testando consultas para aprendizado de inglês...")

    successful_queries = 0
    for query in learning_queries:
        # results = rag_system.query_learning_content(query, n_results=3)
        results = rag_system.enhance_retrieval_for_learning(query, n_results=3)

        if results.get('success') and results['results']:
            successful_queries += 1
            print(f"✅ '{query}': {len(results['results'])} resultados")

            # Mostrar o melhor resultado
            best = results['results'][0]
            print(f"   🎯 [{best['technology']}] {best['content'][:80]}...")
            print(
                f"   📊 Relevância: {best['relevance_score']:.3f} | Inglês: {best['english_level']} | Teaching Score: {best['teaching_score']}")
        else:
            print(f"❌ '{query}': 0 resultados")

    print(f"\n📈 Estatísticas: {successful_queries}/{len(learning_queries)} consultas com resultados")


def create_fallback_content():
    """
    Cria conteúdo de fallback se não houver dados suficientes
    """
    print("\n🔄 Criando conteúdo de fallback para aprendizado...")

    fallback_content = [
        {
            'content': "In programming, a function is a reusable block of code that performs a specific task. Functions help organize code and make it more readable and maintainable. You define a function with a name, parameters, and a body that contains the instructions.",
            'metadata': {
                'technology': 'programming',
                'english_level': 'B1',
                'professional_context': 'development',
                'title': 'Function Definition'
            }
        },
        {
            'content': "Version control systems like Git help developers track changes to their code over time. This allows multiple people to work on the same project without conflicts, and makes it easy to revert changes if something goes wrong.",
            'metadata': {
                'technology': 'git',
                'english_level': 'B1',
                'professional_context': 'collaboration',
                'title': 'Version Control'
            }
        },
        {
            'content': "When writing code documentation, you should explain what the code does, how to use it, and any important considerations. Good documentation helps other developers understand your work and makes maintenance easier.",
            'metadata': {
                'technology': 'documentation',
                'english_level': 'B2',
                'professional_context': 'best_practices',
                'title': 'Code Documentation'
            }
        }
    ]

    print("✅ Conteúdo de fallback criado com 3 exemplos básicos")
    return fallback_content


if __name__ == "__main__":
    demonstrate_enhanced_rag(min_teaching_score=3)