# Synthetic Opinion Lab

A comprehensive framework for generating synthetic public opinion survey data using multiple paradigms including statistical models, Large Language Models (LLMs), and agent-based simulations.

## Overview

The Synthetic Opinion Lab enables researchers to create realistic synthetic survey datasets that preserve the statistical properties and relationships found in real public opinion data. This is useful for:

- Data augmentation and privacy protection
- Testing research methodologies
- Studying opinion dynamics and social influence
- Creating training data for machine learning models

## Key Features

### 🗂️ Multi-Format Survey Ingestion
- Supports SPSS (.sav), Stata (.dta), CSV, and TSV files
- Automatic schema extraction and data cleaning
- Standardized internal representation

### 👥 Realistic Persona Generation
- Demographics based on census distributions (Canada, US)
- Correlated latent traits (ideology, authoritarianism, trust, populism, cosmopolitanism)
- Narrative descriptions for each synthetic respondent

### 🎯 Three Generation Paradigms

1. **Item Response Theory (IRT)**
   - 2PL logistic models for binary items
   - Graded response models for Likert scales
   - Configurable item parameters

2. **Large Language Models**
   - Support for local models (Ollama) and APIs (Together)
   - Persona-driven response generation
   - Adaptive prompt strategies

3. **Agent-Based Simulation**
   - Social network structures with demographic homophily
   - Opinion dynamics with social influence and media effects
   - Realistic opinion evolution over time

### 📊 Comprehensive Evaluation
- Distribution similarity metrics (KL divergence, Jensen-Shannon, Wasserstein distance)
- Correlation structure preservation analysis
- Regression model replication tests
- Automated quality scoring and reporting

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd apopsi

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
from synthetic_opinion_lab import run_simple_experiment

# Run an experiment with IRT generation
results = run_simple_experiment(
    survey_file="my_survey.csv",
    generator_type="irt",
    n_personas=1000
)

print(f"Quality score: {results['evaluation_results']['overall_quality']['summary']['overall_score']}")
```

### Compare All Generators

```python
from synthetic_opinion_lab import compare_all_generators

# Compare IRT, LLM, and agent-based approaches
comparison = compare_all_generators(
    survey_file="my_survey.csv",
    n_personas=500
)

print("Generator ranking:", comparison['comparison_summary']['ranking'])
```

### Custom Workflow

```python
from synthetic_opinion_lab import (
    SurveyIngester, PersonaGenerator, IRTResponseGenerator, EvaluationPipeline
)

# 1. Ingest survey data
real_data, schema = SurveyIngester.ingest("survey.sav")

# 2. Generate personas
persona_gen = PersonaGenerator.default_canadian()
personas = persona_gen.generate(1000)

# 3. Generate responses
irt_gen = IRTResponseGenerator()
synthetic_data = irt_gen.generate(personas, schema)

# 4. Evaluate quality
evaluator = EvaluationPipeline()
results = evaluator.run_full_evaluation(real_data, synthetic_data)
```

## Configuration

### LLM Setup

For LLM-based generation, you'll need either:

**Ollama (Local)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3
```

**Together API (Cloud)**
```bash
export TOGETHER_API_KEY="your-api-key"
```

### Agent-Based Simulation

Configure network structures and dynamics:

```python
config = {
    "n_timesteps": 50,
    "network_type": "small_world",
    "network_params": {"k": 6, "p": 0.1},
    "media_influence_strength": 0.1
}

results = run_simple_experiment(
    survey_file="survey.csv",
    generator_type="agent_simulation",
    generator_config=config
)
```

## Framework Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Real Survey   │───▶│ Survey Ingestion│───▶│Standard Schema  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐             │
                       │Persona Generator│◄────────────┘
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Response      │
                       │   Generators    │
                       │                 │
                       │  ┌───────────┐  │
                       │  │    IRT    │  │
                       │  │    LLM    │  │
                       │  │  Agents   │  │
                       │  └───────────┘  │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │Synthetic Dataset│
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Evaluation    │
                       │   & Reports     │
                       └─────────────────┘
```

## Examples

See `example_usage.py` for detailed examples including:

- Persona generation with custom trait distributions
- IRT response generation with parameter tuning
- LLM-based generation with different prompt templates
- Agent simulation with social networks
- Comprehensive evaluation and visualization

## Advanced Features

### Custom Trait Models

```python
from synthetic_opinion_lab.personas.trait_models import TraitModel
import numpy as np

# Define custom trait correlations
trait_names = ["ideology", "authoritarianism", "trust_in_government", "populism", "cosmopolitanism"]
custom_correlations = np.array([
    [1.0,  -0.4,  0.2,  -0.3,  0.7],
    [-0.4,  1.0,  0.3,   0.5, -0.4],
    [0.2,   0.3,  1.0,  -0.2,  0.1],
    [-0.3,  0.5, -0.2,   1.0, -0.5],
    [0.7,  -0.4,  0.1,  -0.5,  1.0]
])

trait_model = TraitModel(
    trait_names=trait_names,
    means=np.zeros(5),
    covariance_matrix=custom_correlations
)
```

### Batch Processing

```python
from synthetic_opinion_lab.pipelines.survey_replication_pipeline import SurveyReplicationPipeline

pipeline = SurveyReplicationPipeline()

# Process multiple surveys
survey_files = ["survey1.csv", "survey2.sav", "survey3.dta"]
batch_results = pipeline.batch_replicate_surveys(survey_files)
```

## Evaluation Metrics

The framework provides comprehensive evaluation including:

- **Distribution Metrics**: KL divergence, Jensen-Shannon divergence, Total Variation distance
- **Correlation Analysis**: Matrix comparison, preservation of strong correlations
- **Regression Tests**: Coefficient replication, model performance comparison
- **Quality Scoring**: Composite quality scores across all dimensions

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{synthetic_opinion_lab,
  title={Synthetic Opinion Lab: A Framework for Generating Synthetic Public Opinion Survey Data},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Check the documentation and examples
- Open an issue on GitHub
- Review the evaluation reports for quality insights

## Roadmap

Planned features:
- Integration with more LLM providers
- Advanced opinion dynamics models
- Time-series survey replication
- Multi-language survey support
- Automated hyperparameter optimization