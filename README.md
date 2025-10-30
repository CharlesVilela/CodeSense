# CodeSense: AI-Powered English Learning for Developers

## 📚 Project Overview

**CodeSense** is an intelligent Retrieval-Augmented Generation (RAG) system specifically designed to teach technical English to software developers. By leveraging authentic documentation and code repositories, CodeSense creates a contextual learning environment where developers can improve their English skills while engaging with real-world technical content.

## 🎯 Problem Statement

Software developers learning English often struggle with:
- **Technical vocabulary** specific to programming and development
- **Documentation comprehension** in English
- **Professional communication** in development teams
- **Contextual understanding** of technical concepts in English

Traditional English learning materials lack the technical context that developers need, while technical documentation lacks pedagogical structure for language learning.

## ✨ Key Features

### 🧠 Intelligent Content Processing
- **Advanced Text Cleaning**: Removes HTML, markdown, and code while preserving explanatory content
- **Pedagogical Filtering**: Scores content based on teaching value using pattern recognition
- **Context Preservation**: Maintains technical relevance while focusing on language learning

### 🎓 Educational Enhancements
- **Vocabulary Extraction**: Identifies technical terms for focused learning
- **Grammar Pattern Recognition**: Highlights grammatical structures in technical context
- **English Level Classification**: Automatically categorizes content by proficiency (B1, B2, C1)
- **Teaching Quality Scoring**: Rates content from 1-10 based on pedagogical value

### 🔍 Smart Retrieval System
- **Semantic Search**: Combines TF-IDF with teaching quality scores
- **Query Optimization**: Expands and refines learning-focused queries
- **Multi-technology Support**: Covers 20+ technologies including React, AWS, Docker, TypeScript, and more

## 🛠️ Technical Architecture

### Data Processing Pipeline
```
Raw Documentation → Advanced Cleaning → Explanatory Extraction → Pedagogical Filtering → Enhanced Chunks
```

### RAG System Components
- **Content Processor**: `EnhancedRAGDataProcessor` with intelligent chunking
- **Teaching Filter**: `TeachingQualityFilter` with pattern-based scoring
- **Retrieval Engine**: `EnhancedRAGSystem` with semantic + pedagogical ranking
- **Query Optimizer**: Automatic query expansion for learning contexts

## 📊 Current Performance

- **✅ 100% Query Success Rate**: 20/20 test queries return relevant results
- **🎯 High-Quality Content**: 413 pedagogical chunks (from 1,637 total)
- **📈 Teaching Quality**: Average score of 3.51/10 across all content
- **🌍 Technology Coverage**: 20+ technologies with balanced representation
- **🎓 Level Distribution**: 387 B1, 24 B2, 2 C1 level chunks

## 🚀 Getting Started

### Prerequisites
```python
# Core dependencies
Python 3.8+
scikit-learn
numpy
```

### Installation
```bash
git clone https://github.com/your-username/codesense.git
cd codesense
pip install -r requirements.txt
```

### Basic Usage
```python
from codesense import EnhancedRAGDataProcessor, EnhancedRAGSystem

# Initialize and process content
processor = EnhancedRAGDataProcessor()
learning_chunks = processor.process_for_english_learning()

# Create RAG system
rag_system = EnhancedRAGSystem(learning_chunks)

# Query for learning content
results = rag_system.query_learning_content(
    "How to define a function in programming?",
    n_results=3
)
```

## 📁 Project Structure
```
codesense/
├── processors/          # Content processing modules
├── filters/            # Pedagogical filtering
├── retrieval/          # RAG system components
├── data/              # Technical documentation
├── outputs/           # Processed learning chunks
└── tests/             # Test suites
```

## 🎯 Use Cases

### For Developers
- **Technical Vocabulary Building**: Learn programming terms in context
- **Documentation Reading Practice**: Improve comprehension of technical docs
- **Professional Communication**: Study how concepts are explained in English
- **Interview Preparation**: Practice technical explanations in English

### For Educational Institutions
- **Custom Learning Paths**: Technology-specific English courses
- **Curriculum Development**: Authentic technical content for language classes
- **Assessment Tools**: Evaluate technical English proficiency

## 🔮 Future Roadmap

### Short-term Goals
- [ ] Web interface for interactive learning
- [ ] Expanded technology coverage
- [ ] User progress tracking
- [ ] Exercise generation from content

### Long-term Vision
- [ ] Multi-modal learning (code + explanation)
- [ ] Personalized learning paths
- [ ] Collaborative learning features
- [ ] Integration with development environments

## 🤝 Contributing

We welcome contributions from:
- **Developers** interested in educational technology
- **Educators** with experience in technical English
- **Linguists** specializing in English for Specific Purposes (ESP)
- **Technical Writers** with documentation expertise

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏆 Acknowledgments

- Technical documentation from AWS, Google Cloud, React, and other open-source projects
- Educational research in English for Specific Purposes (ESP)
- Open-source NLP libraries that make this project possible

---

**CodeSense**: Bridging the gap between technical expertise and English proficiency, one developer at a time. 🚀
